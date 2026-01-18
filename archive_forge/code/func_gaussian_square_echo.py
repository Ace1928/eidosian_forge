from __future__ import annotations
import functools
import warnings
from collections.abc import Mapping, Callable
from copy import deepcopy
from typing import Any
import numpy as np
import symengine as sym
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform
def gaussian_square_echo(duration: int | ParameterValueType, amp: float | ParameterExpression, sigma: float | ParameterExpression, width: float | ParameterExpression | None=None, angle: float | ParameterExpression | None=0.0, active_amp: float | ParameterExpression | None=0.0, active_angle: float | ParameterExpression | None=0.0, risefall_sigma_ratio: float | ParameterExpression | None=None, name: str | None=None, limit_amplitude: bool | None=None) -> SymbolicPulse:
    """An echoed Gaussian square pulse with an active tone overlaid on it.

    The Gaussian Square Echo pulse is composed of three pulses. First, a Gaussian Square pulse
    :math:`f_{echo}(x)` with amplitude ``amp`` and phase ``angle`` playing for half duration,
    followed by a second Gaussian Square pulse :math:`-f_{echo}(x)` with opposite amplitude
    and same phase playing for the rest of the duration. Third a Gaussian Square pulse
    :math:`f_{active}(x)` with amplitude ``active_amp`` and phase ``active_angle``
    playing for the entire duration. The Gaussian Square Echo pulse :math:`g_e()`
    can be written as:

    .. math::

        \\begin{aligned}
        g_e(x) &= \\begin{cases}            f_{\\text{active}} + f_{\\text{echo}}(x)                & x < \\frac{\\text{duration}}{2}\\\\
            f_{\\text{active}} - f_{\\text{echo}}(x)                & \\frac{\\text{duration}}{2} < x        \\end{cases}\\\\
        \\end{aligned}

    One case where this pulse can be used is when implementing a direct CNOT gate with
    a cross-resonance superconducting qubit architecture. When applying this pulse to
    the target qubit, the active portion can be used to cancel IX terms from the
    cross-resonance drive while the echo portion can reduce the impact of a static ZZ coupling.

    Exactly one of the ``risefall_sigma_ratio`` and ``width`` parameters has to be specified.

    If ``risefall_sigma_ratio`` is not ``None`` and ``width`` is ``None``:

    .. math::

        \\begin{aligned}
        \\text{risefall} &= \\text{risefall\\_sigma\\_ratio} \\times \\text{sigma}\\\\
        \\text{width} &= \\text{duration} - 2 \\times \\text{risefall}
        \\end{aligned}

    If ``width`` is not None and ``risefall_sigma_ratio`` is None:

    .. math:: \\text{risefall} = \\frac{\\text{duration} - \\text{width}}{2}

    References:
        1. |citation1|_

        .. _citation1: https://iopscience.iop.org/article/10.1088/2058-9565/abe519

        .. |citation1| replace:: *Jurcevic, P., Javadi-Abhari, A., Bishop, L. S.,
            Lauer, I., Bogorin, D. F., Brink, M., Capelluto, L., G{"u}nl{"u}k, O.,
            Itoko, T., Kanazawa, N. & others
            Demonstration of quantum volume 64 on a superconducting quantum
            computing system. (Section V)*
    Args:
        duration: Pulse length in terms of the sampling period `dt`.
        amp: The amplitude of the rise and fall and of the echoed pulse.
        sigma: A measure of how wide or narrow the risefall is; see the class
               docstring for more details.
        width: The duration of the embedded square pulse.
        angle: The angle in radians of the complex phase factor uniformly
            scaling the echoed pulse. Default value 0.
        active_amp: The amplitude of the active pulse.
        active_angle: The angle in radian of the complex phase factor uniformly
            scaling the active pulse. Default value 0.
        risefall_sigma_ratio: The ratio of each risefall duration to sigma.
        name: Display name for this pulse envelope.
        limit_amplitude: If ``True``, then limit the amplitude of the
            waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

    Returns:
        ScalableSymbolicPulse instance.
    Raises:
        PulseError: When width and risefall_sigma_ratio are both empty or both non-empty.
    """
    if width is None and risefall_sigma_ratio is None:
        raise PulseError('Either the pulse width or the risefall_sigma_ratio parameter must be specified.')
    if width is not None and risefall_sigma_ratio is not None:
        raise PulseError('Either the pulse width or the risefall_sigma_ratio parameter can be specified but not both.')
    if width is None and risefall_sigma_ratio is not None:
        width = duration - 2.0 * risefall_sigma_ratio * sigma
    parameters = {'amp': amp, 'angle': angle, 'sigma': sigma, 'width': width, 'active_amp': active_amp, 'active_angle': active_angle}
    _t, _duration, _amp, _sigma, _active_amp, _width, _angle, _active_angle = sym.symbols('t, duration, amp, sigma, active_amp, width, angle, active_angle')
    _center = _duration / 4
    _width_echo = (_duration - 2 * (_duration - _width)) / 2
    _sq_t0 = _center - _width_echo / 2
    _sq_t1 = _center + _width_echo / 2
    _gaussian_ledge = _lifted_gaussian(_t, _sq_t0, -1, _sigma)
    _gaussian_redge = _lifted_gaussian(_t, _sq_t1, _duration / 2 + 1, _sigma)
    envelope_expr_p = _amp * sym.exp(sym.I * _angle) * sym.Piecewise((_gaussian_ledge, _t <= _sq_t0), (_gaussian_redge, _t >= _sq_t1), (1, True))
    _center_echo = _duration / 2 + _duration / 4
    _sq_t0_echo = _center_echo - _width_echo / 2
    _sq_t1_echo = _center_echo + _width_echo / 2
    _gaussian_ledge_echo = _lifted_gaussian(_t, _sq_t0_echo, _duration / 2 - 1, _sigma)
    _gaussian_redge_echo = _lifted_gaussian(_t, _sq_t1_echo, _duration + 1, _sigma)
    envelope_expr_echo = -1 * _amp * sym.exp(sym.I * _angle) * sym.Piecewise((_gaussian_ledge_echo, _t <= _sq_t0_echo), (_gaussian_redge_echo, _t >= _sq_t1_echo), (1, True))
    envelope_expr = sym.Piecewise((envelope_expr_p, _t <= _duration / 2), (envelope_expr_echo, _t >= _duration / 2), (0, True))
    _center_active = _duration / 2
    _sq_t0_active = _center_active - _width / 2
    _sq_t1_active = _center_active + _width / 2
    _gaussian_ledge_active = _lifted_gaussian(_t, _sq_t0_active, -1, _sigma)
    _gaussian_redge_active = _lifted_gaussian(_t, _sq_t1_active, _duration + 1, _sigma)
    envelope_expr_active = _active_amp * sym.exp(sym.I * _active_angle) * sym.Piecewise((_gaussian_ledge_active, _t <= _sq_t0_active), (_gaussian_redge_active, _t >= _sq_t1_active), (1, True))
    envelop_expr_total = envelope_expr + envelope_expr_active
    consts_expr = sym.And(_sigma > 0, _width >= 0, _duration >= _width, _duration / 2 >= _width_echo)
    valid_amp_conditions_expr = sym.And(sym.Abs(_amp) + sym.Abs(_active_amp) <= 1.0)
    return SymbolicPulse(pulse_type='gaussian_square_echo', duration=duration, parameters=parameters, name=name, limit_amplitude=limit_amplitude, envelope=envelop_expr_total, constraints=consts_expr, valid_amp_conditions=valid_amp_conditions_expr)
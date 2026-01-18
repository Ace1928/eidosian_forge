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
def get_waveform(self) -> Waveform:
    """Return a Waveform with samples filled according to the formula that the pulse
        represents and the parameter values it contains.

        Since the returned array is a discretized time series of the continuous function,
        this method uses a midpoint sampler. For ``duration``, return:

        .. math::

            \\{f(t+0.5) \\in \\mathbb{C} | t \\in \\mathbb{Z} \\wedge  0<=t<\\texttt{duration}\\}

        Returns:
            A waveform representation of this pulse.

        Raises:
            PulseError: When parameters are not assigned.
            PulseError: When expression for pulse envelope is not assigned.
        """
    if self.is_parameterized():
        raise PulseError('Unassigned parameter exists. All parameters must be assigned.')
    if self._envelope is None:
        raise PulseError('Pulse envelope expression is not assigned.')
    fargs = _get_expression_args(self._envelope, self.parameters)
    return Waveform(samples=self._envelope_lam(*fargs), name=self.name)
import enum
from typing import Any, List, Optional, TYPE_CHECKING, Union
import pandas as pd
import sympy
from matplotlib import pyplot as plt
from cirq import circuits, ops, study, value
from cirq._compat import proper_repr
def t2_decay(sampler: 'cirq.Sampler', *, qubit: 'cirq.Qid', experiment_type: 'ExperimentType'=ExperimentType.RAMSEY, num_points: int, max_delay: 'cirq.DURATION_LIKE', min_delay: 'cirq.DURATION_LIKE'=None, repetitions: int=1000, delay_sweep: Optional[study.Sweep]=None, num_pulses: Optional[List[int]]=None) -> Union['cirq.experiments.T2DecayResult', List['cirq.experiments.T2DecayResult']]:
    """Runs a t2 transverse relaxation experiment.

    Initializes a qubit into a superposition state, evolves the system using
    rules determined by the experiment type and by the delay parameters,
    then rotates back for measurement.  This will measure the phase decoherence
    decay.  This experiment has three types of T2 metrics, each which measure
    a different slice of the noise spectrum.

    For the Ramsey experiment type (often denoted T2*), the state will be
    prepared with a square root Y gate (`cirq.Y ** 0.5`) and then waits for
    a variable amount of time.  After this time, it will do basic state
    tomography to measure the expectation of the Pauli-X and Pauli-Y operators
    by performing either a `cirq.Y ** -0.5` or `cirq.X ** 0.5`.  The square of
    these two measurements is summed to determine the length of the Bloch
    vector. This experiment measures the phase decoherence of the system under
    free evolution.

    For the Hahn echo experiment (often denoted T2 or spin echo), the state
    will also be prepared with a square root Y gate (`cirq.Y ** 0.5`).
    However, during the mid-point of the delay time being measured, a pi-pulse
    (`cirq.X`) gate will be applied to cancel out inhomogeneous dephasing.
    The same method of measuring the final state as Ramsey experiment is applied
    after the second half of the delay period.  See the animation on the wiki
    page https://en.wikipedia.org/wiki/Spin_echo for a visual illustration
    of this experiment.

    CPMG, or the Carr-Purcell-Meiboom-Gill sequence, involves using a sqrt(Y)
    followed by a sequence of pi pulses (X gates) in a specific timing pattern:

        π/2, t, π, 2t, π, ... 2t, π, t

    The first pulse, a sqrt(Y) gate, will put the qubit's state on the Bloch
    equator.  After a delay, successive X gates will refocus dehomogenous
    phase effects by causing them to precess in opposite directions and
    averaging their effects across the entire pulse train.

    This pulse pattern has two variables that can be adjusted.  The first,
    denoted as 't' in the above sequence, is delay, which can be specified
    with `delay_min` and `delay_max` or by using a `delay_sweep`, similar to
    the other experiments.  The second variable is the number of pi pulses
    (X gates).  This can be specified as a list of integers using the
    `num_pulses` parameter.  If multiple different pulses are specified,
    the data will be presented in a data frame with two
    indices (delay_ns and num_pulses).

    See the following reference for more information about CPMG pulse trains:
    Meiboom, S., and D. Gill, “Modified spin-echo method for measuring nuclear
    relaxation times”, Rev. Sci. Inst., 29, 688–691 (1958).
    https://doi.org/10.1063/1.1716296

    Note that interpreting T2 data is fairly tricky and subtle, as it can
    include other effects that need to be accounted for.  For instance,
    amplitude damping (T1) will present as T2 noise and needs to be
    appropriately compensated for to find a true measure of T2.  Due to this
    subtlety and lack of standard way to interpret the data, the fitting
    of the data to an exponential curve and the extrapolation of an actual
    T2 time value is left as an exercise to the reader.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        experiment_type: The type of T2 test to run.
        num_points: The number of evenly spaced delays to test.
        max_delay: The largest delay to test.
        min_delay: The smallest delay to test. Defaults to no delay.
        repetitions: The number of repetitions of the circuit
             for each delay and for each tomography result.
        delay_sweep: Optional range of time delays to sweep across.  This should
             be a SingleSweep using the 'delay_ns' with values in integer number
             of nanoseconds.  If specified, this will override the max_delay and
             min_delay parameters.  If not specified, the experiment will sweep
             from min_delay to max_delay with linear steps.
        num_pulses: For CPMG, a list of the number of pulses to use.
             If multiple pulses are specified, each will be swept on.

    Returns:
        A T2DecayResult object that stores and can plot the data.

    Raises:
        ValueError: If invalid parameters are specified, negative repetitions, max
            less than min durations, negative min delays, or an unsupported experiment
            configuraiton.
    """
    min_delay_dur = value.Duration(min_delay)
    max_delay_dur = value.Duration(max_delay)
    min_delay_nanos = min_delay_dur.total_nanos()
    max_delay_nanos = max_delay_dur.total_nanos()
    if repetitions <= 0:
        raise ValueError('repetitions <= 0')
    if isinstance(min_delay_nanos, sympy.Expr) or isinstance(max_delay_nanos, sympy.Expr):
        raise ValueError('min_delay and max_delay cannot be sympy expressions.')
    if max_delay_dur < min_delay_dur:
        raise ValueError('max_delay < min_delay')
    if min_delay_dur < 0:
        raise ValueError('min_delay < 0')
    if num_pulses and experiment_type != ExperimentType.CPMG:
        raise ValueError('num_pulses is only valid for CPMG experiments.')
    delay_var = sympy.Symbol('delay_ns')
    inv_x_var = sympy.Symbol('inv_x')
    inv_y_var = sympy.Symbol('inv_y')
    max_pulses = max(num_pulses) if num_pulses else 0
    if not delay_sweep:
        delay_sweep = study.Linspace(delay_var, start=min_delay_nanos, stop=max_delay_nanos, length=num_points)
    if delay_sweep.keys != ['delay_ns']:
        raise ValueError('delay_sweep must be a SingleSweep with delay_ns parameter')
    if experiment_type == ExperimentType.RAMSEY:
        circuit = circuits.Circuit(ops.Y(qubit) ** 0.5, ops.wait(qubit, nanos=delay_var))
    else:
        if experiment_type == ExperimentType.HAHN_ECHO:
            num_pulses = [0]
        if not num_pulses:
            raise ValueError('At least one value must be given for num_pulses in a CPMG experiment')
        circuit = _cpmg_circuit(qubit, delay_var, max_pulses)
    circuit.append(ops.X(qubit) ** inv_x_var)
    circuit.append(ops.Y(qubit) ** inv_y_var)
    circuit.append(ops.measure(qubit, key='output'))
    tomography_sweep = study.Zip(study.Points('inv_x', [0.0, 0.5]), study.Points('inv_y', [-0.5, 0.0]))
    if num_pulses and max_pulses > 0:
        pulse_sweep = _cpmg_sweep(num_pulses)
        sweep = study.Product(delay_sweep, pulse_sweep, tomography_sweep)
    else:
        sweep = study.Product(delay_sweep, tomography_sweep)
    results = sampler.sample(circuit, params=sweep, repetitions=repetitions)
    y_basis_measurements = results[abs(results.inv_y) > 0].copy()
    x_basis_measurements = results[abs(results.inv_x) > 0].copy()
    if num_pulses and len(num_pulses) > 1:
        cols = tuple((f'pulse_{t}' for t in range(max_pulses)))
        x_basis_measurements['num_pulses'] = x_basis_measurements.loc[:, cols].sum(axis=1)
        y_basis_measurements['num_pulses'] = y_basis_measurements.loc[:, cols].sum(axis=1)
    x_basis_tabulation = _create_tabulation(x_basis_measurements)
    y_basis_tabulation = _create_tabulation(y_basis_measurements)
    return T2DecayResult(x_basis_tabulation, y_basis_tabulation)
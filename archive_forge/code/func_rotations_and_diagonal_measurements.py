import copy
from threading import RLock
import pennylane as qml
from pennylane.measurements import CountsMP, ProbabilityMP, SampleMP, MeasurementProcess
from pennylane.operation import DecompositionUndefinedError, Operator, StatePrepBase
from pennylane.queuing import AnnotatedQueue, QueuingManager, process_queue
from pennylane.pytrees import register_pytree
from .qscript import QuantumScript
def rotations_and_diagonal_measurements(tape):
    """Compute the rotations for overlapping observables, and return them along with the diagonalized observables."""
    if not tape._obs_sharing_wires:
        return ([], tape.measurements)
    with QueuingManager.stop_recording():
        try:
            rotations, diag_obs = qml.pauli.diagonalize_qwc_pauli_words(tape._obs_sharing_wires)
        except (TypeError, ValueError) as e:
            if any((isinstance(m, (ProbabilityMP, SampleMP, CountsMP)) for m in tape.measurements)):
                raise qml.QuantumFunctionError('Only observables that are qubit-wise commuting Pauli words can be returned on the same wire.\nTry removing all probability, sample and counts measurements this will allow for splitting of execution and separate measurements for each non-commuting observable.') from e
            raise qml.QuantumFunctionError(_err_msg_for_some_meas_not_qwc(tape.measurements)) from e
        measurements = copy.copy(tape.measurements)
        for o, i in zip(diag_obs, tape._obs_sharing_wires_id):
            new_m = tape.measurements[i].__class__(obs=o)
            measurements[i] = new_m
    return (rotations, measurements)
import copy
from threading import RLock
import pennylane as qml
from pennylane.measurements import CountsMP, ProbabilityMP, SampleMP, MeasurementProcess
from pennylane.operation import DecompositionUndefinedError, Operator, StatePrepBase
from pennylane.queuing import AnnotatedQueue, QueuingManager, process_queue
from pennylane.pytrees import register_pytree
from .qscript import QuantumScript
def expand_tape(tape, depth=1, stop_at=None, expand_measurements=False):
    """Expand all objects in a tape to a specific depth.

    Args:
        tape (QuantumTape): The tape to expand
        depth (int): the depth the tape should be expanded
        stop_at (Callable): A function which accepts a queue object,
            and returns ``True`` if this object should *not* be expanded.
            If not provided, all objects that support expansion will be expanded.
        expand_measurements (bool): If ``True``, measurements will be expanded
            to basis rotations and computational basis measurements.

    Returns:
        QuantumTape: The expanded version of ``tape``.

    Raises:
        QuantumFunctionError: if some observables in the tape are not qubit-wise commuting

    **Example**

    Consider the following nested tape:

    .. code-block:: python

        with QuantumTape() as tape:
            qml.BasisState(np.array([1, 1]), wires=[0, 'a'])

            with QuantumTape() as tape2:
                qml.Rot(0.543, 0.1, 0.4, wires=0)

            qml.CNOT(wires=[0, 'a'])
            qml.RY(0.2, wires='a')
            qml.probs(wires=0), qml.probs(wires='a')

    The nested structure is preserved:

    >>> tape.operations
    [BasisState(array([1, 1]), wires=[0, 'a']),
     <QuantumTape: wires=[0], params=3>,
     CNOT(wires=[0, 'a']),
     RY(0.2, wires=['a'])]

    Calling ``expand_tape`` will return a tape with all nested tapes
    expanded, resulting in a single tape of quantum operations:

    >>> new_tape = qml.tape.tape.expand_tape(tape)
    >>> new_tape.operations
    [BasisStatePreparation([1, 1], wires=[0, 'a']),
    Rot(0.543, 0.1, 0.4, wires=[0]),
    CNOT(wires=[0, 'a']),
    RY(0.2, wires=['a'])]
    """
    if depth == 0:
        return tape
    if stop_at is None:

        def stop_at(obj):
            return False
    new_ops = []
    new_measurements = []
    if tape.samples_computational_basis and len(tape.measurements) > 1:
        _validate_computational_basis_sampling(tape.measurements)
    diagonalizing_gates, diagonal_measurements = rotations_and_diagonal_measurements(tape)
    for queue, new_queue in [(tape.operations + diagonalizing_gates, new_ops), (diagonal_measurements, new_measurements)]:
        for obj in queue:
            stop_at_meas = not expand_measurements and isinstance(obj, MeasurementProcess)
            if stop_at_meas or stop_at(obj):
                new_queue.append(obj)
                continue
            if isinstance(obj, Operator):
                if obj.has_decomposition:
                    with QueuingManager.stop_recording():
                        obj = QuantumScript(obj.decomposition(), _update=False)
                else:
                    new_queue.append(obj)
                    continue
            elif isinstance(obj, qml.measurements.MeasurementProcess):
                try:
                    obj = obj.expand()
                except DecompositionUndefinedError:
                    new_queue.append(obj)
                    continue
            expanded_tape = expand_tape(obj, stop_at=stop_at, depth=depth - 1)
            new_ops.extend(expanded_tape.operations)
            new_measurements.extend(expanded_tape.measurements)
    new_tape = tape.__class__(new_ops, new_measurements, shots=tape.shots, _update=False)
    new_tape.wires = copy.copy(tape.wires)
    new_tape.num_wires = tape.num_wires
    new_tape._batch_size = tape._batch_size
    new_tape._output_dim = tape._output_dim
    return new_tape
import contextlib
import pennylane as qml
from pennylane.measurements import Shots
@contextlib.contextmanager
def set_shots(device, shots):
    """Context manager to temporarily change the shots
    of a device.

    This context manager can be used in two ways.

    As a standard context manager:

    >>> dev = qml.device("default.qubit.legacy", wires=2, shots=None)
    >>> with set_shots(dev, shots=100):
    ...     print(dev.shots)
    100
    >>> print(dev.shots)
    None

    Or as a decorator that acts on a function that uses the device:

    >>> set_shots(dev, shots=100)(lambda: dev.shots)()
    100
    """
    if isinstance(device, qml.devices.Device):
        raise ValueError('The new device interface is not compatible with `set_shots`. Set shots when calling the qnode or put the shots on the QuantumTape.')
    if isinstance(shots, Shots):
        shots = shots.shot_vector if shots.has_partitioned_shots else shots.total_shots
    if shots == device.shots:
        yield
        return
    original_shots = device.shots
    original_shot_vector = device._shot_vector
    try:
        if shots is not False and device.shots != shots:
            device.shots = shots
        yield
    finally:
        device.shots = original_shots
        device._shot_vector = original_shot_vector
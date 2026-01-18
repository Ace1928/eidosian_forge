import functools
import itertools
import numbers
import warnings
import numpy as np
from scipy.linalg import solve as linalg_solve
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.functions import bind_new_parameters
from pennylane.tape import QuantumScript
def generate_shifted_tapes(tape, index, shifts, multipliers=None, broadcast=False):
    """Generate a list of tapes or a single broadcasted tape, where one marked
    trainable parameter has been shifted by the provided shift values.

    Args:
        tape (.QuantumTape): input quantum tape
        index (int): index of the trainable parameter to shift
        shifts (Sequence[float or int]): sequence of shift values.
            The length determines how many parameter-shifted tapes are created.
        multipliers (Sequence[float or int]): sequence of multiplier values.
            The length should match the one of ``shifts``. Each multiplier scales the
            corresponding gate parameter before the shift is applied. If not provided, the
            parameters will not be scaled.
        broadcast (bool): Whether or not to use broadcasting to create a single tape
            with the shifted parameters.

    Returns:
        list[QuantumTape]: List of quantum tapes. In each tape the parameter indicated
            by ``index`` has been shifted by the values in ``shifts``. The number of tapes
            matches the length of ``shifts`` and ``multipliers`` (if provided).
            If ``broadcast=True`` was used, the list contains a single broadcasted tape
            with all shifts distributed over the broadcasting dimension. In this case,
            the ``batch_size`` of the returned tape matches the length of ``shifts``.
    """
    if multipliers is None:
        multipliers = np.ones_like(shifts)
    if broadcast:
        return (_copy_and_shift_params(tape, [index], [shifts], [multipliers]),)
    return tuple((_copy_and_shift_params(tape, [index], [shift], [multiplier]) for shift, multiplier in zip(shifts, multipliers)))
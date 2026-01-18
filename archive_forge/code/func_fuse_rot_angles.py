from functools import partial
import numpy as np
from pennylane.math import abs as math_abs
from pennylane.math import sum as math_sum
from pennylane.math import allclose, arccos, arctan2, cos, get_interface, is_abstract, sin, stack
from pennylane.wires import Wires
from pennylane.ops.identity import GlobalPhase
def fuse_rot_angles(angles_1, angles_2):
    """Computed the set of rotation angles that is obtained when composing
    two ``qml.Rot`` operations.

    The ``qml.Rot`` operation represents the most general single-qubit operation.
    Two such operations can be fused into a new operation, however the angular dependence
    is non-trivial.

    Args:
        angles_1 (float): A set of three angles for the first ``qml.Rot`` operation.
        angles_2 (float): A set of three angles for the second ``qml.Rot`` operation.

    Returns:
        array[float]: Rotation angles for a single ``qml.Rot`` operation that
        implements the same operation as the two sets of input angles.
    """
    if is_abstract(angles_1) or is_abstract(angles_2):
        interface = get_interface(angles_1, angles_2)
        if interface == 'jax':
            from jax.lax import cond
            return cond(allclose(angles_1[1], 0.0) * allclose(angles_2[1], 0.0), _no_fuse, partial(_fuse, abstract_jax=True), angles_1, angles_2)
    if allclose(angles_1[1], 0.0) and allclose(angles_2[1], 0.0):
        return _no_fuse(angles_1, angles_2)
    return _fuse(angles_1, angles_2)
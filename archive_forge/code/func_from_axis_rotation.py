from __future__ import annotations
import math
import numpy as np
@classmethod
def from_axis_rotation(cls, angle: float, axis: str) -> Quaternion:
    """Return quaternion for rotation about given axis.

        Args:
            angle (float): Angle in radians.
            axis (str): Axis for rotation

        Returns:
            Quaternion: Quaternion for axis rotation.

        Raises:
            ValueError: Invalid input axis.
        """
    out = np.zeros(4, dtype=float)
    if axis == 'x':
        out[1] = 1
    elif axis == 'y':
        out[2] = 1
    elif axis == 'z':
        out[3] = 1
    else:
        raise ValueError('Invalid axis input.')
    out *= math.sin(angle / 2.0)
    out[0] = math.cos(angle / 2.0)
    return cls(out)
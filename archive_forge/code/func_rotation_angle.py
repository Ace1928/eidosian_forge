from collections import namedtuple
import math
import warnings
@property
def rotation_angle(self) -> float:
    """The rotation angle in degrees of the affine transformation.

        This is the rotation angle in degrees of the affine transformation,
        assuming it is in the form M = R S, where R is a rotation and S is a
        scaling.

        Raises UndefinedRotationError for improper and degenerate
        transformations.
        """
    a, b, _, c, d, _, _, _, _ = self
    if self.is_proper or self.is_degenerate:
        l1, _ = self._scaling
        y, x = (c / l1, a / l1)
        return math.atan2(y, x) * 180 / math.pi
    else:
        raise UndefinedRotationError
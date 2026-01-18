from collections import namedtuple
import math
import warnings
def itransform(self, seq) -> None:
    """Transform a sequence of points or vectors in place.

        :param seq: Mutable sequence of :class:`~planar.Vec2` to be
            transformed.
        :returns: None, the input sequence is mutated in place.
        """
    if self is not identity and self != identity:
        sa, sb, sc, sd, se, sf, _, _, _ = self
        for i, (x, y) in enumerate(seq):
            seq[i] = (x * sa + y * sb + sc, x * sd + y * se + sf)
import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
def get_rigid_rotation(self, dstpts):
    """
        Extract the rotation to apply to a group of points to minimize the
        distance to a second group of points. The two groups of points are
        assumed to be centered. This is a simple version that just picks
        an angle based on the first point of the gesture.
        """
    if len(self.strokes) < 1 or len(self.strokes[0].points) < 1:
        return 0
    if len(dstpts.strokes) < 1 or len(dstpts.strokes[0].points) < 1:
        return 0
    p = dstpts.strokes[0].points[0]
    target = Vector([p.x, p.y])
    source = Vector([p.x, p.y])
    return source.angle(target)
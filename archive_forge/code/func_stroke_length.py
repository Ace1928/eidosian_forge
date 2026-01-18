import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
def stroke_length(self, point_list=None):
    """Finds the length of the stroke. If a point list is given,
           finds the length of that list.
        """
    if point_list is None:
        point_list = self.points
    gesture_length = 0.0
    if len(point_list) <= 1:
        return gesture_length
    for i in range(len(point_list) - 1):
        gesture_length += self.points_distance(point_list[i], point_list[i + 1])
    return gesture_length
import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
def normalize_stroke(self, sample_points=32):
    """Normalizes strokes so that every stroke has a standard number of
           points. Returns True if stroke is normalized, False if it can't be
           normalized. sample_points controls the resolution of the stroke.
        """
    if len(self.points) <= 1 or self.stroke_length(self.points) == 0.0:
        return False
    target_stroke_size = self.stroke_length(self.points) / float(sample_points)
    new_points = list()
    new_points.append(self.points[0])
    prev = self.points[0]
    src_distance = 0.0
    dst_distance = target_stroke_size
    for curr in self.points[1:]:
        d = self.points_distance(prev, curr)
        if d > 0:
            prev = curr
            src_distance = src_distance + d
            while dst_distance < src_distance:
                x_dir = curr.x - prev.x
                y_dir = curr.y - prev.y
                ratio = (src_distance - dst_distance) / d
                to_x = x_dir * ratio + prev.x
                to_y = y_dir * ratio + prev.y
                new_points.append(GesturePoint(to_x, to_y))
                dst_distance = self.stroke_length(self.points) / float(sample_points) * len(new_points)
    if not len(new_points) == sample_points:
        raise ValueError('Invalid number of strokes points; got %d while it should be %d' % (len(new_points), sample_points))
    self.points = new_points
    return True
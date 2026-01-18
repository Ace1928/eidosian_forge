import unittest
from kivy.vector import Vector
from operator import truediv
def test_intersection_roundingerror(self):
    v1 = (25.0, 200.0)
    v2 = (25.0, 400.0)
    v3 = (36.75, 300.0)
    result = [25.0, 300.0]

    def almost(a, b):
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertAlmostEqual(a[0], b[0], places=0)
        self.assertAlmostEqual(a[1], b[1], places=0)
    for i in range(1, 100):
        st = '6.4' + '9' * i
        v = (float(st), 300.0)
        almost(result, Vector.segment_intersection(v1, v2, v3, v))
    for i in range(1, 100):
        st = '6.1' + '1' * i
        v = (float(st), 300.0)
        almost(result, Vector.segment_intersection(v1, v2, v3, v))
    for i in range(1, 100):
        st = '6.4' + '4' * i
        v = (float(st), 300.0)
        almost(result, Vector.segment_intersection(v1, v2, v3, v))
    for i in range(1, 100):
        st = '300.4' + '9' * i
        v = (6.5, float(st))
        almost(result, Vector.segment_intersection(v1, v2, v3, v))
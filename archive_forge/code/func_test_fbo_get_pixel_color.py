import unittest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.widget import Widget
from kivy.graphics import Fbo, Color, Rectangle
def test_fbo_get_pixel_color(self):
    fbow = FboTest()
    render_error = 2
    values = [(tuple, int, (0, 0, 0, 0)), (list, int, [0, 0, 0, 0]), (list, int, [0, 72, 0, 128]), (list, int, [72, 0, 0, 128]), (list, int, [36, 72, 0, 255]), (list, int, [0, 145, 0, 255]), (list, int, [145, 0, 0, 255]), (list, int, [0, 145, 0, 255])]
    for i, pos in enumerate(fbow.positions):
        c = fbow.fbo.get_pixel_color(pos[0], pos[1])
        self.assertTrue(isinstance(c, values[i][0]))
        for v in c:
            self.assertTrue(isinstance(v, values[i][1]))
        for j, val in enumerate(c):
            self.assertAlmostEqual(val, values[i][2][j], delta=render_error)
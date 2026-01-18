import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_fbo_pixels(self):
    from kivy.graphics import Fbo, ClearColor, ClearBuffers, Ellipse
    fbo = Fbo(size=(512, 512))
    with fbo:
        ClearColor(0, 0, 0, 1)
        ClearBuffers()
        Ellipse(pos=(100, 100), size=(100, 100))
    fbo.draw()
    data = fbo.pixels
    fbo.texture.save('results.png')
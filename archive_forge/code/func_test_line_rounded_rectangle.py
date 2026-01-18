import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
def test_line_rounded_rectangle(self):
    from kivy.uix.widget import Widget
    from kivy.graphics import Line, Color
    r = self.render
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1)
        line = Line(rounded_rectangle=(100, 100, 100, 100, 10, 20, 30, 40, 100))
    r(wid)
    assert line.rounded_rectangle == (100, 100, 100, 100, 10, 20, 30, 40, 100)
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1)
        line = Line(rounded_rectangle=(100, 100, 100, 100, 100, 20, 10, 30, 100))
    r(wid)
    assert line.rounded_rectangle == (100, 100, 100, 100, 70, 20, 10, 30, 100)
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1)
        line = Line(rounded_rectangle=(100, 100, 100, 100, 100, 25, 100, 50, 100))
    r(wid)
    assert line.rounded_rectangle == (100, 100, 100, 100, 50, 25, 50, 50, 100)
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1)
        line = Line(rounded_rectangle=(100, 100, 100, 100, 150, 50, 50.001, 51, 100))
    r(wid)
    assert line.rounded_rectangle == (100, 100, 100, 100, 50, 50, 50, 50, 100)
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1)
        line = Line(rounded_rectangle=(100, 100, 100, 100, 0, 0, 0, 0, 100))
    r(wid)
    assert line.rounded_rectangle == (100, 100, 100, 100, 1, 1, 1, 1, 100)
    wid = Widget()
    with wid.canvas:
        Color(1, 1, 1)
        line = Line(rounded_rectangle=(100, 100, 100, 100, 100, 0, 0, 0, 100))
    r(wid)
    assert line.rounded_rectangle == (100, 100, 100, 100, 99, 1, 1, 1, 100)
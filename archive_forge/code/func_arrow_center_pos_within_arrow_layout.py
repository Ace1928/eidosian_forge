import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
@property
def arrow_center_pos_within_arrow_layout(self):
    x = self._arrow_image_scatter_wrapper.center_x
    y = self._arrow_image_scatter_wrapper.center_y
    return (x, y)
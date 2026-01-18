from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.properties import (
def stop_cursor(self, instance, mouse_pos):
    self.offset_x = 0
    self.offset_y = 0
    self.pos = (mouse_pos[0] - self.size[0] / 2.0, mouse_pos[1] - self.size[1] / 2.0)
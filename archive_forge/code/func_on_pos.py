from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.properties import (
def on_pos(self, instance, new_pos):
    self.set_cursor()
    self.cursor_x.points = self.cursor_pts[:4]
    self.cursor_y.points = self.cursor_pts[4:]
    self.cursor_ox.points = self.cursor_pts[:4]
    self.cursor_oy.points = self.cursor_pts[4:]
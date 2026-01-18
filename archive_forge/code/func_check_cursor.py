from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.properties import (
def check_cursor(self, win, stickid, axisid, value):
    intensity = self.intensity
    dead = self.dead_zone
    if axisid == 3:
        if value < -dead:
            self.offset_x = -intensity
        elif value > dead:
            self.offset_x = intensity
        else:
            self.offset_x = 0
    elif axisid == 4:
        if value < -dead:
            self.offset_y = intensity
        elif value > dead:
            self.offset_y = -intensity
        else:
            self.offset_y = 0
    else:
        self.offset_x = 0
        self.offset_y = 0
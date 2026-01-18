from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.properties import (
def joystick_shortcuts(self, win, stickid, buttonid):
    if buttonid == 7:
        self.activated = not self.activated
        if self.activated:
            self.pos = [round(i / 2.0) for i in win.size]
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.properties import (
def keyboard_shortcuts(self, win, scancode, *args):
    modifiers = args[-1]
    if scancode == 101 and modifiers == ['ctrl']:
        self.activated = not self.activated
        return True
    elif scancode == 27:
        if self.activated:
            self.activated = False
            return True
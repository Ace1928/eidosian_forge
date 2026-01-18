from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.properties import (
def set_intensity(self, win, stickid, buttonid):
    intensity = self.intensity
    if buttonid == 0 and intensity > 2:
        intensity -= 1
    elif buttonid == 1:
        intensity += 1
    self.intensity = intensity
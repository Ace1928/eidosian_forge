from time import time
from kivy.effects.kinetic import KineticEffect
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ObjectProperty
def on_value(self, *args):
    scroll_min = self.min
    scroll_max = self.max
    if scroll_min > scroll_max:
        scroll_min, scroll_max = (scroll_max, scroll_min)
    if self.value < scroll_min:
        self.overscroll = self.value - scroll_min
        self.reset(scroll_min)
    elif self.value > scroll_max:
        self.overscroll = self.value - scroll_max
        self.reset(scroll_max)
    else:
        self.scroll = self.value
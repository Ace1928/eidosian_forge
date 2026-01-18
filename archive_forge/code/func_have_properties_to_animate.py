from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
def have_properties_to_animate(self, widget):
    return self.anim1.have_properties_to_animate(widget) or self.anim2.have_properties_to_animate(widget)
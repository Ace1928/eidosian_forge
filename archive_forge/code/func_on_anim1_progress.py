from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
def on_anim1_progress(self, instance, widget, progress):
    self.dispatch('on_progress', widget, progress / 2.0)
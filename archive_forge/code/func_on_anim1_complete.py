from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
def on_anim1_complete(self, instance, widget):
    if widget.uid not in self._widgets:
        return
    self.anim2.start(widget)
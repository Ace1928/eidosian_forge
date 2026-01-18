from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
def on_anim2_complete(self, instance, widget):
    """Repeating logic used with boolean variable "repeat".

        .. versionadded:: 1.7.1
        """
    if widget.uid not in self._widgets:
        return
    if self.repeat:
        self.anim1.start(widget)
    else:
        self.dispatch('on_complete', widget)
        self.cancel(widget)
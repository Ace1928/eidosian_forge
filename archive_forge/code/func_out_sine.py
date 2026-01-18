from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def out_sine(progress):
    """.. image:: images/anim_out_sine.png
        """
    return sin(progress * (pi / 2.0))
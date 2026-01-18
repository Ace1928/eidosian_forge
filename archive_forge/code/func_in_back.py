from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def in_back(progress):
    """.. image:: images/anim_in_back.png
        """
    return progress * progress * ((1.70158 + 1.0) * progress - 1.70158)
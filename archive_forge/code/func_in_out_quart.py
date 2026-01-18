from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def in_out_quart(progress):
    """.. image:: images/anim_in_out_quart.png
        """
    p = progress * 2
    if p < 1:
        return 0.5 * p * p * p * p
    p -= 2
    return -0.5 * (p * p * p * p - 2.0)
from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def in_out_expo(progress):
    """.. image:: images/anim_in_out_expo.png
        """
    if progress == 0:
        return 0.0
    if progress == 1.0:
        return 1.0
    p = progress * 2
    if p < 1:
        return 0.5 * pow(2, 10 * (p - 1.0))
    p -= 1.0
    return 0.5 * (-pow(2, -10 * p) + 2.0)
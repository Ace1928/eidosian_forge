from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
@staticmethod
def in_out_bounce(progress):
    """.. image:: images/anim_in_out_bounce.png
        """
    p = progress * 2.0
    if p < 1.0:
        return AnimationTransition._in_bounce_internal(p, 1.0) * 0.5
    return AnimationTransition._out_bounce_internal(p - 1.0, 1.0) * 0.5 + 0.5
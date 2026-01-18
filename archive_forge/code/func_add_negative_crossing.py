from ..sage_helper import _within_sage
from . import exhaust
from .links_base import Link
def add_negative_crossing(self, i):
    return self.cap_then_cup(i) + -q * self
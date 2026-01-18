from ..sage_helper import _within_sage
from . import exhaust
from .links_base import Link
def add_positive_crossing(self, i):
    return self + -q * self.cap_then_cup(i)
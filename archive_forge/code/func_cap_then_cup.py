from ..sage_helper import _within_sage
from . import exhaust
from .links_base import Link
def cap_then_cup(self, i):
    return self.cap_off(i).insert_cup(i)
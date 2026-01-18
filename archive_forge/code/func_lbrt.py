from param.parameterized import get_occupied_slots
from .util import datetime_types
def lbrt(self):
    """Return (left,bottom,right,top) as a tuple."""
    return (self._left, self._bottom, self._right, self._top)
import functools
import operator
import math
import datetime as DT
from matplotlib import _api
from matplotlib.dates import date2num
def julianDate(self, frame):
    t = self
    if frame != self._frame:
        t = self.convert(frame)
    return t._jd + t._seconds / 86400.0
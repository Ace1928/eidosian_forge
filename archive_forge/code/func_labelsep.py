import math
import numpy as np
from numpy import ma
from matplotlib import _api, cbook, _docstring
import matplotlib.artist as martist
import matplotlib.collections as mcollections
from matplotlib.patches import CirclePolygon
import matplotlib.text as mtext
import matplotlib.transforms as transforms
@property
def labelsep(self):
    return self._labelsep_inches * self.Q.axes.figure.dpi
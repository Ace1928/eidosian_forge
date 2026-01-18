import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
def is_first_col(self):
    return self.colspan.start == 0
import math
import numpy as np
import matplotlib as mpl
from matplotlib.patches import _Style, FancyArrowPatch
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform
def set_line_mutation_scale(self, scale):
    self.set_mutation_scale(scale * self._line_mutation_scale)
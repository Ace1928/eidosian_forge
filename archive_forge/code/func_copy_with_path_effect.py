from matplotlib.backend_bases import RendererBase
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms
from matplotlib.path import Path
import numpy as np
def copy_with_path_effect(self, path_effects):
    return self.__class__(path_effects, self._renderer)
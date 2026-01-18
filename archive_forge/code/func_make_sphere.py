import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Patch3D
from .utils import matplotlib_close_if_inline
def make_sphere(self):
    """
        Plots Bloch sphere and data sets.
        """
    self.render()
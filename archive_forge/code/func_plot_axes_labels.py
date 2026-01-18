import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Patch3D
from .utils import matplotlib_close_if_inline
def plot_axes_labels(self):
    """axes labels"""
    opts = {'fontsize': self.font_size, 'color': self.font_color, 'horizontalalignment': 'center', 'verticalalignment': 'center'}
    self.axes.text(0, -self.xlpos[0], 0, self.xlabel[0], **opts)
    self.axes.text(0, -self.xlpos[1], 0, self.xlabel[1], **opts)
    self.axes.text(self.ylpos[0], 0, 0, self.ylabel[0], **opts)
    self.axes.text(self.ylpos[1], 0, 0, self.ylabel[1], **opts)
    self.axes.text(0, 0, self.zlpos[0], self.zlabel[0], **opts)
    self.axes.text(0, 0, self.zlpos[1], self.zlabel[1], **opts)
    for item in self.axes.xaxis.get_ticklines() + self.axes.xaxis.get_ticklabels():
        item.set_visible(False)
    for item in self.axes.yaxis.get_ticklines() + self.axes.yaxis.get_ticklabels():
        item.set_visible(False)
    for item in self.axes.zaxis.get_ticklines() + self.axes.zaxis.get_ticklabels():
        item.set_visible(False)
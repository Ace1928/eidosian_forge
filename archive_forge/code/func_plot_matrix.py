import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
def plot_matrix(self, colorbar_kws, xind, yind, **kws):
    self.data2d = self.data2d.iloc[yind, xind]
    self.mask = self.mask.iloc[yind, xind]
    xtl = kws.pop('xticklabels', 'auto')
    try:
        xtl = np.asarray(xtl)[xind]
    except (TypeError, IndexError):
        pass
    ytl = kws.pop('yticklabels', 'auto')
    try:
        ytl = np.asarray(ytl)[yind]
    except (TypeError, IndexError):
        pass
    annot = kws.pop('annot', None)
    if annot is None or annot is False:
        pass
    else:
        if isinstance(annot, bool):
            annot_data = self.data2d
        else:
            annot_data = np.asarray(annot)
            if annot_data.shape != self.data2d.shape:
                err = '`data` and `annot` must have same shape.'
                raise ValueError(err)
            annot_data = annot_data[yind][:, xind]
        annot = annot_data
    kws.setdefault('cbar', self.ax_cbar is not None)
    heatmap(self.data2d, ax=self.ax_heatmap, cbar_ax=self.ax_cbar, cbar_kws=colorbar_kws, mask=self.mask, xticklabels=xtl, yticklabels=ytl, annot=annot, **kws)
    ytl = self.ax_heatmap.get_yticklabels()
    ytl_rot = None if not ytl else ytl[0].get_rotation()
    self.ax_heatmap.yaxis.set_ticks_position('right')
    self.ax_heatmap.yaxis.set_label_position('right')
    if ytl_rot is not None:
        ytl = self.ax_heatmap.get_yticklabels()
        plt.setp(ytl, rotation=ytl_rot)
    tight_params = dict(h_pad=0.02, w_pad=0.02)
    if self.ax_cbar is None:
        self._figure.tight_layout(**tight_params)
    else:
        self.ax_cbar.set_axis_off()
        self._figure.tight_layout(**tight_params)
        self.ax_cbar.set_axis_on()
        self.ax_cbar.set_position(self.cbar_pos)
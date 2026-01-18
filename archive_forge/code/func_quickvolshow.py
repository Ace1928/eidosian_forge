from __future__ import absolute_import
import logging
import warnings
import numpy as np
import ipywidgets as widgets  # we should not have widgets under two names
import ipywebrtc
import pythreejs
import traitlets
from traitlets import Unicode, Integer
from traittypes import Array
from bqplot import scales
import ipyvolume
import ipyvolume as ipv  # we should not have ipyvolume under two names either
import ipyvolume._version
from ipyvolume.traittypes import Image
from ipyvolume.serialize import (
from ipyvolume.transferfunction import TransferFunction
from ipyvolume.utils import debounced, grid_slice, reduce_size
def quickvolshow(data, lighting=False, data_min=None, data_max=None, max_shape=256, level=[0.1, 0.5, 0.9], opacity=[0.01, 0.05, 0.1], level_width=0.1, extent=None, memorder='C', **kwargs):
    """Visualize a 3d array using volume rendering.

    :param data: 3d numpy array
    :param lighting: boolean, to use lighting or not, if set to false, lighting parameters will be overriden
    :param data_min: minimum value to consider for data, if None, computed using np.nanmin
    :param data_max: maximum value to consider for data, if None, computed using np.nanmax
    :param int max_shape: maximum shape for the 3d cube, if larger, the data is reduced by skipping/slicing (data[::N]),
                          set to None to disable.
    :param extent: list of [[xmin, xmax], [ymin, ymax], [zmin, zmax]] values that define the bounds of the volume,
                   otherwise the viewport is used
    :param level: level(s) for the where the opacity in the volume peaks, maximum sequence of length 3
    :param opacity: opacity(ies) for each level, scalar or sequence of max length 3
    :param level_width: width of the (gaussian) bumps where the opacity peaks, scalar or sequence of max length 3
    :param kwargs: extra argument passed to Volume and default transfer function
    :return:

    """
    ipv.figure()
    ipv.volshow(data, lighting=lighting, data_min=data_min, data_max=data_max, max_shape=max_shape, level=level, opacity=opacity, level_width=level_width, extent=extent, memorder=memorder, **kwargs)
    return ipv.gcc()
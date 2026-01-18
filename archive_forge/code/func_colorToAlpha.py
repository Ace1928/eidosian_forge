from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def colorToAlpha(data, color):
    """
    Given an RGBA image in *data*, convert *color* to be transparent. 
    *data* must be an array (w, h, 3 or 4) of ubyte values and *color* must be 
    an array (3) of ubyte values.
    This is particularly useful for use with images that have a black or white background.
    
    Algorithm is taken from Gimp's color-to-alpha function in plug-ins/common/colortoalpha.c
    Credit:
        /*
        * Color To Alpha plug-in v1.0 by Seth Burgess, sjburges@gimp.org 1999/05/14
        *  with algorithm by clahey
        */
    
    """
    data = data.astype(float)
    if data.shape[-1] == 3:
        d2 = np.empty(data.shape[:2] + (4,), dtype=data.dtype)
        d2[..., :3] = data
        d2[..., 3] = 255
        data = d2
    color = color.astype(float)
    alpha = np.zeros(data.shape[:2] + (3,), dtype=float)
    output = data.copy()
    for i in [0, 1, 2]:
        d = data[..., i]
        c = color[i]
        mask = d > c
        alpha[..., i][mask] = (d[mask] - c) / (255.0 - c)
        imask = d < c
        alpha[..., i][imask] = (c - d[imask]) / c
    output[..., 3] = alpha.max(axis=2) * 255.0
    mask = output[..., 3] >= 1.0
    correction = 255.0 / output[..., 3][mask]
    for i in [0, 1, 2]:
        output[..., i][mask] = (output[..., i][mask] - color[i]) * correction + color[i]
        output[..., 3][mask] *= data[..., 3][mask] / 255.0
    return np.clip(output, 0, 255).astype(np.ubyte)
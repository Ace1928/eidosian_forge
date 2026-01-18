import ctypes
import logging
import os
import shutil
import warnings
import numpy as np
from ..base import _LIB, check_call, py_str
from ..base import c_array, c_str, mx_uint, c_str_array
from ..base import NDArrayHandle, SymbolHandle
from ..symbol import Symbol
from ..symbol import load as sym_load
from .. import ndarray
from ..ndarray import load as nd_load
from ..ndarray import save as nd_save
from ..ndarray import NDArray
from ..io import DataIter, DataDesc, DataBatch
from ..context import cpu, Context
from ..module import Module
def combine_histogram(old_hist, arr, new_min, new_max, new_th):
    """ Collect layer histogram for arr and combine it with old histogram.
    """
    old_hist, old_hist_edges, old_min, old_max, old_th = old_hist
    if new_th <= old_th:
        hist, _ = np.histogram(arr, bins=len(old_hist), range=(-old_th, old_th))
        return (old_hist + hist, old_hist_edges, min(old_min, new_min), max(old_max, new_max), old_th)
    else:
        old_num_bins = len(old_hist)
        old_step = 2 * old_th / old_num_bins
        half_increased_bins = int((new_th - old_th) // old_step + 1)
        new_num_bins = half_increased_bins * 2 + old_num_bins
        new_th = half_increased_bins * old_step + old_th
        hist, hist_edges = np.histogram(arr, bins=new_num_bins, range=(-new_th, new_th))
        hist[half_increased_bins:new_num_bins - half_increased_bins] += old_hist
        return (hist, hist_edges, min(old_min, new_min), max(old_max, new_max), new_th)
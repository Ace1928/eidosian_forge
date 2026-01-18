import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
def merge_csvs(in_list):
    for idx, in_file in enumerate(in_list):
        try:
            in_array = np.loadtxt(in_file, delimiter=',')
        except ValueError:
            try:
                in_array = np.loadtxt(in_file, delimiter=',', skiprows=1)
            except ValueError:
                with open(in_file, 'r') as first:
                    header_line = first.readline()
                header_list = header_line.split(',')
                n_cols = len(header_list)
                try:
                    in_array = np.loadtxt(in_file, delimiter=',', skiprows=1, usecols=list(range(1, n_cols)))
                except ValueError:
                    in_array = np.loadtxt(in_file, delimiter=',', skiprows=1, usecols=list(range(1, n_cols - 1)))
        if idx == 0:
            out_array = in_array
        else:
            out_array = np.dstack((out_array, in_array))
    out_array = np.squeeze(out_array)
    iflogger.info('Final output array shape:')
    iflogger.info(np.shape(out_array))
    return out_array
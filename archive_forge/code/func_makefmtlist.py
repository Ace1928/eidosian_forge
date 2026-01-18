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
def makefmtlist(output_array, typelist, rowheadingsBool, shape, extraheadingBool):
    fmtlist = []
    if rowheadingsBool:
        fmtlist.append('%s')
    if len(shape) > 1:
        output = np.zeros(max(shape), typelist)
        for idx in range(1, min(shape) + 1):
            output[str(idx)] = output_array[:, idx - 1]
            fmtlist.append('%f')
    else:
        output = np.zeros(1, typelist)
        for idx in range(1, len(output_array) + 1):
            output[str(idx)] = output_array[idx - 1]
            fmtlist.append('%f')
    if extraheadingBool:
        fmtlist.append('%s')
    fmt = ','.join(fmtlist)
    return (fmt, output)
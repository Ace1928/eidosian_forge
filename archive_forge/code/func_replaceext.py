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
def replaceext(in_list, ext):
    out_list = list()
    for filename in in_list:
        path, name, _ = split_filename(op.abspath(filename))
        out_name = op.join(path, name) + ext
        out_list.append(out_name)
    return out_list
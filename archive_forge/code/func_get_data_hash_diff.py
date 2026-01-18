import hashlib
import os
import re
import sys
from collections import OrderedDict
from optparse import Option, OptionParser
import numpy as np
import nibabel as nib
import nibabel.cmdline.utils
def get_data_hash_diff(files, dtype=np.float64):
    """Get difference between md5 values of data

    Parameters
    ----------
    files: list of actual files

    Returns
    -------
    list
      np.array: md5 values of respective files
    """
    md5sums = [hashlib.md5(np.ascontiguousarray(nib.load(f).get_fdata(dtype=dtype))).hexdigest() for f in files]
    if len(set(md5sums)) == 1:
        return []
    return md5sums
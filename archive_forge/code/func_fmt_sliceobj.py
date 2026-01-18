import gc
import itertools as it
from timeit import timeit
from unittest import mock
import numpy as np
import nibabel as nib
from nibabel.openers import HAVE_INDEXED_GZIP
from nibabel.tmpdirs import InTemporaryDirectory
from ..rstutils import rst_table
from .butils import print_git_title
def fmt_sliceobj(sliceobj):
    slcstr = []
    for i, s in enumerate(sliceobj):
        if s in ':?':
            slcstr.append(s)
        else:
            slcstr.append(str(int(s * SHAPE[i])))
    return f'[{', '.join(slcstr)}]'
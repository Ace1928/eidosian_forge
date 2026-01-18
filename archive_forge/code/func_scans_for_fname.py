import os
from copy import deepcopy
from nibabel import load
import numpy as np
from ... import logging
from ...utils import spm_docs as sd
from ..base import (
from ..base.traits_extension import NoDefaultSpecified
from ..matlab import MatlabCommand
from ...external.due import due, Doi, BibTeX
def scans_for_fname(fname):
    """Reads a nifti file and converts it to a numpy array storing
    individual nifti volumes.

    Opens images so will fail if they are not found.

    """
    if isinstance(fname, list):
        scans = np.zeros((len(fname),), dtype=object)
        for sno, f in enumerate(fname):
            scans[sno] = '%s,1' % f
        return scans
    img = load(fname)
    if len(img.shape) == 3:
        return np.array(('%s,1' % fname,), dtype=object)
    else:
        n_scans = img.shape[3]
        scans = np.zeros((n_scans,), dtype=object)
        for sno in range(n_scans):
            scans[sno] = '%s,%d' % (fname, sno + 1)
        return scans
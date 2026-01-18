import getpass
import time
import warnings
from collections import OrderedDict
import numpy as np
from ..openers import Opener
def read_morph_data(filepath):
    """Read a Freesurfer morphometry data file.

    This function reads in what Freesurfer internally calls "curv" file types,
    (e.g. ?h. curv, ?h.thickness), but as that has the potential to cause
    confusion where "curv" also refers to the surface curvature values,
    we refer to these files as "morphometry" files with PySurfer.

    Parameters
    ----------
    filepath : str
        Path to morphometry file

    Returns
    -------
    curv : numpy array
        Vector representation of surface morpometry values
    """
    with open(filepath, 'rb') as fobj:
        magic = _fread3(fobj)
        if magic == 16777215:
            vnum = np.fromfile(fobj, '>i4', 3)[0]
            curv = np.fromfile(fobj, '>f4', vnum)
        else:
            vnum = magic
            _fread3(fobj)
            curv = np.fromfile(fobj, '>i2', vnum) / 100
    return curv
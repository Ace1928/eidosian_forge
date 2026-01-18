import getpass
import time
import warnings
from collections import OrderedDict
import numpy as np
from ..openers import Opener
def read_label(filepath, read_scalars=False):
    """Load in a Freesurfer .label file.

    Parameters
    ----------
    filepath : str
        Path to label file.
    read_scalars : bool, optional
        If True, read and return scalars associated with each vertex.

    Returns
    -------
    label_array : numpy array
        Array with indices of vertices included in label.
    scalar_array : numpy array (floats)
        Only returned if `read_scalars` is True.  Array of scalar data for each
        vertex.
    """
    label_array = np.loadtxt(filepath, dtype=int, skiprows=2, usecols=[0])
    if read_scalars:
        scalar_array = np.loadtxt(filepath, skiprows=2, usecols=[-1])
        return (label_array, scalar_array)
    return label_array
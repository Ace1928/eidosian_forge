from .utils import check_output, where
import os
import warnings
import numpy as np
def getLibAVPath():
    """ Returns the path to the directory containing both avconv and avprobe
    """
    return _AVCONV_PATH
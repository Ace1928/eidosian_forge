from .utils import check_output, where
import os
import warnings
import numpy as np
def getLibAVVersion():
    """ Returns the version of LibAV that is currently being used
    """
    return '%s.%s' % (_LIBAV_MAJOR_VERSION, _LIBAV_MINOR_VERSION)
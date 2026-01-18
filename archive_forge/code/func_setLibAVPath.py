from .utils import check_output, where
import os
import warnings
import numpy as np
def setLibAVPath(path):
    """ Sets up the path to the directory containing both avconv and avprobe

        Use this function for to specify specific system installs of LibAV. All
        calls to avconv and avprobe will use this path as a prefix.

        Parameters
        ----------
        path : string
            Path to directory containing avconv and avprobe

        Returns
        -------
        none

    """
    global _AVCONV_PATH
    global _HAS_AVCONV
    _AVCONV_PATH = path
    if os.path.isfile(os.path.join(_AVCONV_PATH, _AVCONV_APPLICATION)) and os.path.isfile(os.path.join(_AVCONV_PATH, _AVPROBE_APPLICATION)):
        _HAS_AVCONV = 1
    else:
        warnings.warn('avconv/avprobe not found in path: ' + str(path), UserWarning)
        _HAS_AVCONV = 0
        global _LIBAV_MAJOR_VERSION
        global _LIBAV_MINOR_VERSION
        _LIBAV_MAJOR_VERSION = '0'
        _LIBAV_MINOR_VERSION = '0'
        return
    scan_libav()
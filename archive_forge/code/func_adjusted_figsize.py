import abc
import base64
import contextlib
from io import BytesIO, TextIOWrapper
import itertools
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import uuid
import warnings
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib._animation_data import (
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
def adjusted_figsize(w, h, dpi, n):
    """
    Compute figure size so that pixels are a multiple of n.

    Parameters
    ----------
    w, h : float
        Size in inches.

    dpi : float
        The dpi.

    n : int
        The target multiple.

    Returns
    -------
    wnew, hnew : float
        The new figure size in inches.
    """

    def correct_roundoff(x, dpi, n):
        if int(x * dpi) % n != 0:
            if int(np.nextafter(x, np.inf) * dpi) % n == 0:
                x = np.nextafter(x, np.inf)
            elif int(np.nextafter(x, -np.inf) * dpi) % n == 0:
                x = np.nextafter(x, -np.inf)
        return x
    wnew = int(w * dpi / n) * n / dpi
    hnew = int(h * dpi / n) * n / dpi
    return (correct_roundoff(wnew, dpi, n), correct_roundoff(hnew, dpi, n))
import atexit
import functools
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
import weakref
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook
from matplotlib.testing.exceptions import ImageComparisonFailure
def calculate_rms(expected_image, actual_image):
    """
    Calculate the per-pixel errors, then compute the root mean square error.
    """
    if expected_image.shape != actual_image.shape:
        raise ImageComparisonFailure(f'Image sizes do not match expected size: {expected_image.shape} actual size {actual_image.shape}')
    return np.sqrt(((expected_image - actual_image).astype(float) ** 2).mean())
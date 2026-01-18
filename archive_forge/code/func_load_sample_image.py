import csv
import gzip
import hashlib
import os
import shutil
from collections import namedtuple
from importlib import resources
from numbers import Integral
from os import environ, listdir, makedirs
from os.path import expanduser, isdir, join, splitext
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
from ..preprocessing import scale
from ..utils import Bunch, check_pandas_support, check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
@validate_params({'image_name': [StrOptions({'china.jpg', 'flower.jpg'})]}, prefer_skip_nested_validation=True)
def load_sample_image(image_name):
    """Load the numpy array of a single sample image.

    Read more in the :ref:`User Guide <sample_images>`.

    Parameters
    ----------
    image_name : {`china.jpg`, `flower.jpg`}
        The name of the sample image loaded.

    Returns
    -------
    img : 3D array
        The image as a numpy array: height x width x color.

    Examples
    --------

    >>> from sklearn.datasets import load_sample_image
    >>> china = load_sample_image('china.jpg')   # doctest: +SKIP
    >>> china.dtype                              # doctest: +SKIP
    dtype('uint8')
    >>> china.shape                              # doctest: +SKIP
    (427, 640, 3)
    >>> flower = load_sample_image('flower.jpg') # doctest: +SKIP
    >>> flower.dtype                             # doctest: +SKIP
    dtype('uint8')
    >>> flower.shape                             # doctest: +SKIP
    (427, 640, 3)
    """
    images = load_sample_images()
    index = None
    for i, filename in enumerate(images.filenames):
        if filename.endswith(image_name):
            index = i
            break
    if index is None:
        raise AttributeError('Cannot find sample image: %s' % image_name)
    return images.images[index]
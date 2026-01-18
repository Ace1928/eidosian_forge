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
def load_sample_images():
    """Load sample images for image manipulation.

    Loads both, ``china`` and ``flower``.

    Read more in the :ref:`User Guide <sample_images>`.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        images : list of ndarray of shape (427, 640, 3)
            The two sample image.
        filenames : list
            The filenames for the images.
        DESCR : str
            The full description of the dataset.

    Examples
    --------
    To load the data and visualize the images:

    >>> from sklearn.datasets import load_sample_images
    >>> dataset = load_sample_images()     #doctest: +SKIP
    >>> len(dataset.images)                #doctest: +SKIP
    2
    >>> first_img_data = dataset.images[0] #doctest: +SKIP
    >>> first_img_data.shape               #doctest: +SKIP
    (427, 640, 3)
    >>> first_img_data.dtype               #doctest: +SKIP
    dtype('uint8')
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError('The Python Imaging Library (PIL) is required to load data from jpeg files. Please refer to https://pillow.readthedocs.io/en/stable/installation.html for installing PIL.')
    descr = load_descr('README.txt', descr_module=IMAGES_MODULE)
    filenames, images = ([], [])
    jpg_paths = sorted((resource for resource in resources.files(IMAGES_MODULE).iterdir() if resource.is_file() and resource.match('*.jpg')))
    for path in jpg_paths:
        filenames.append(str(path))
        with path.open('rb') as image_file:
            pil_image = Image.open(image_file)
            image = np.asarray(pil_image)
        images.append(image)
    return Bunch(images=images, filenames=filenames, DESCR=descr)
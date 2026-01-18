import os
import shutil
import tempfile
import warnings
from functools import partial
from importlib import resources
from pathlib import Path
from pickle import dumps, loads
import numpy as np
import pytest
from sklearn.datasets import (
from sklearn.datasets._base import (
from sklearn.datasets.tests.test_common import check_as_frame
from sklearn.preprocessing import scale
from sklearn.utils import Bunch
def test_load_sample_images():
    try:
        res = load_sample_images()
        assert len(res.images) == 2
        assert len(res.filenames) == 2
        images = res.images
        assert np.all(images[0][0, 0, :] == np.array([174, 201, 231], dtype=np.uint8))
        assert np.all(images[1][0, 0, :] == np.array([2, 19, 13], dtype=np.uint8))
        assert res.DESCR
    except ImportError:
        warnings.warn('Could not load sample images, PIL is not available.')
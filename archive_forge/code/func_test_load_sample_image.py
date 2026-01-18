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
def test_load_sample_image():
    try:
        china = load_sample_image('china.jpg')
        assert china.dtype == 'uint8'
        assert china.shape == (427, 640, 3)
    except ImportError:
        warnings.warn('Could not load sample images, PIL is not available.')
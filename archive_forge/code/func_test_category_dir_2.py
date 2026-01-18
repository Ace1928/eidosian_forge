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
@pytest.fixture
def test_category_dir_2(load_files_root):
    test_category_dir2 = tempfile.mkdtemp(dir=load_files_root)
    yield str(test_category_dir2)
    _remove_dir(test_category_dir2)
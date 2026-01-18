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
def test_load_files_wo_load_content(test_category_dir_1, test_category_dir_2, load_files_root):
    res = load_files(load_files_root, load_content=False)
    assert len(res.filenames) == 1
    assert len(res.target_names) == 2
    assert res.DESCR is None
    assert res.get('data') is None
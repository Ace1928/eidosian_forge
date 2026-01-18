import os
import random
import shutil
import tempfile
from functools import partial
import numpy as np
import pytest
from sklearn.datasets import fetch_lfw_pairs, fetch_lfw_people
from sklearn.datasets.tests.test_common import check_return_X_y
from sklearn.utils._testing import assert_array_equal
def test_load_fake_lfw_people_too_restrictive():
    with pytest.raises(ValueError):
        fetch_lfw_people(data_home=SCIKIT_LEARN_DATA, min_faces_per_person=100, download_if_missing=False)
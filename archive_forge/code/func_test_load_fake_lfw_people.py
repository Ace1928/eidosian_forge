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
def test_load_fake_lfw_people():
    lfw_people = fetch_lfw_people(data_home=SCIKIT_LEARN_DATA, min_faces_per_person=3, download_if_missing=False)
    assert lfw_people.images.shape == (10, 62, 47)
    assert lfw_people.data.shape == (10, 2914)
    assert_array_equal(lfw_people.target, [2, 0, 1, 0, 2, 0, 2, 1, 1, 2])
    expected_classes = ['Abdelatif Smith', 'Abhati Kepler', 'Onur Lopez']
    assert_array_equal(lfw_people.target_names, expected_classes)
    lfw_people = fetch_lfw_people(data_home=SCIKIT_LEARN_DATA, resize=None, slice_=None, color=True, download_if_missing=False)
    assert lfw_people.images.shape == (17, 250, 250, 3)
    assert lfw_people.DESCR.startswith('.. _labeled_faces_in_the_wild_dataset:')
    assert_array_equal(lfw_people.target, [0, 0, 1, 6, 5, 6, 3, 6, 0, 3, 6, 1, 2, 4, 5, 1, 2])
    assert_array_equal(lfw_people.target_names, ['Abdelatif Smith', 'Abhati Kepler', 'Camara Alvaro', 'Chen Dupont', 'John Lee', 'Lin Bauman', 'Onur Lopez'])
    fetch_func = partial(fetch_lfw_people, data_home=SCIKIT_LEARN_DATA, resize=None, slice_=None, color=True, download_if_missing=False)
    check_return_X_y(lfw_people, fetch_func)
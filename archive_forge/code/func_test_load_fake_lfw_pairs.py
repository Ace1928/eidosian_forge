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
def test_load_fake_lfw_pairs():
    lfw_pairs_train = fetch_lfw_pairs(data_home=SCIKIT_LEARN_DATA, download_if_missing=False)
    assert lfw_pairs_train.pairs.shape == (10, 2, 62, 47)
    assert_array_equal(lfw_pairs_train.target, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    expected_classes = ['Different persons', 'Same person']
    assert_array_equal(lfw_pairs_train.target_names, expected_classes)
    lfw_pairs_train = fetch_lfw_pairs(data_home=SCIKIT_LEARN_DATA, resize=None, slice_=None, color=True, download_if_missing=False)
    assert lfw_pairs_train.pairs.shape == (10, 2, 250, 250, 3)
    assert_array_equal(lfw_pairs_train.target, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert_array_equal(lfw_pairs_train.target_names, expected_classes)
    assert lfw_pairs_train.DESCR.startswith('.. _labeled_faces_in_the_wild_dataset:')
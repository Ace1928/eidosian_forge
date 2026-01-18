import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import available_if
@pytest.mark.xfail(_IS_WASM, reason='memmap not fully supported')
def test_create_memmap_backed_data(monkeypatch):
    registration_counter = RegistrationCounter()
    monkeypatch.setattr(atexit, 'register', registration_counter)
    input_array = np.ones(3)
    data = create_memmap_backed_data(input_array)
    check_memmap(input_array, data)
    assert registration_counter.nb_calls == 1
    data, folder = create_memmap_backed_data(input_array, return_folder=True)
    check_memmap(input_array, data)
    assert folder == os.path.dirname(data.filename)
    assert registration_counter.nb_calls == 2
    mmap_mode = 'r+'
    data = create_memmap_backed_data(input_array, mmap_mode=mmap_mode)
    check_memmap(input_array, data, mmap_mode)
    assert registration_counter.nb_calls == 3
    input_list = [input_array, input_array + 1, input_array + 2]
    mmap_data_list = create_memmap_backed_data(input_list)
    for input_array, data in zip(input_list, mmap_data_list):
        check_memmap(input_array, data)
    assert registration_counter.nb_calls == 4
    output_data, other = create_memmap_backed_data([input_array, 'not-an-array'])
    check_memmap(input_array, output_data)
    assert other == 'not-an-array'
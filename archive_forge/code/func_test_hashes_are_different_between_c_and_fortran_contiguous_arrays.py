import time
import hashlib
import sys
import gc
import io
import collections
import itertools
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from joblib.hashing import hash
from joblib.func_inspect import filter_args
from joblib.memory import Memory
from joblib.testing import raises, skipif, fixture, parametrize
from joblib.test.common import np, with_numpy
@with_numpy
def test_hashes_are_different_between_c_and_fortran_contiguous_arrays():
    rng = np.random.RandomState(0)
    arr_c = rng.random_sample((10, 10))
    arr_f = np.asfortranarray(arr_c)
    assert hash(arr_c) != hash(arr_f)
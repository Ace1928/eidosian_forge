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
@parametrize('coerce_mmap', [True, False])
def test_hash_memmap(tmpdir, coerce_mmap):
    """Check that memmap and arrays hash identically if coerce_mmap is True."""
    filename = tmpdir.join('memmap_temp').strpath
    try:
        m = np.memmap(filename, shape=(10, 10), mode='w+')
        a = np.asarray(m)
        are_hashes_equal = hash(a, coerce_mmap=coerce_mmap) == hash(m, coerce_mmap=coerce_mmap)
        assert are_hashes_equal == coerce_mmap
    finally:
        if 'm' in locals():
            del m
            gc.collect()
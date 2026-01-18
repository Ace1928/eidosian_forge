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
def test_hashes_stay_the_same_with_numpy_objects():

    def create_objects_to_hash():
        rng = np.random.RandomState(42)
        to_hash_list = [rng.randint(-1000, high=1000, size=50).astype('<i8'), tuple((rng.randn(3).astype('<f4') for _ in range(5))), [rng.randn(3).astype('<f4') for _ in range(5)], {-3333: rng.randn(3, 5).astype('<f4'), 0: [rng.randint(10, size=20).astype('<i8'), rng.randn(10).astype('<f4')]}, np.arange(100, dtype='<i8').reshape((10, 10)), np.asfortranarray(np.arange(100, dtype='<i8').reshape((10, 10))), np.arange(100, dtype='<i8').reshape((10, 10))[:, :2]]
        return to_hash_list
    to_hash_list_one = create_objects_to_hash()
    to_hash_list_two = create_objects_to_hash()
    e1 = ProcessPoolExecutor(max_workers=1)
    e2 = ProcessPoolExecutor(max_workers=1)
    try:
        for obj_1, obj_2 in zip(to_hash_list_one, to_hash_list_two):
            hash_1 = e1.submit(hash, obj_1).result()
            hash_2 = e2.submit(hash, obj_1).result()
            assert hash_1 == hash_2
            hash_3 = e1.submit(hash, obj_2).result()
            assert hash_1 == hash_3
    finally:
        e1.shutdown()
        e2.shutdown()
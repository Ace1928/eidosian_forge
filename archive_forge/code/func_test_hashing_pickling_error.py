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
def test_hashing_pickling_error():

    def non_picklable():
        return 42
    with raises(pickle.PicklingError) as excinfo:
        hash(non_picklable)
    excinfo.match('PicklingError while hashing')
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
def test_bound_cached_methods_hash(tmpdir):
    """ Make sure that calling the same _cached_ method on two different
    instances of the same class does resolve to the same hashes.
    """
    a = KlassWithCachedMethod(tmpdir.strpath)
    b = KlassWithCachedMethod(tmpdir.strpath)
    assert hash(filter_args(a.f.func, [], (1,))) == hash(filter_args(b.f.func, [], (1,)))
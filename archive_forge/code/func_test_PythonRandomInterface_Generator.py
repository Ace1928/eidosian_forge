import random
from copy import copy
import pytest
import networkx as nx
from networkx.utils import (
from networkx.utils.misc import _dict_to_numpy_array1, _dict_to_numpy_array2
def test_PythonRandomInterface_Generator():
    np = pytest.importorskip('numpy')
    rng = np.random.default_rng(42)
    pri = PythonRandomInterface(np.random.default_rng(42))
    assert pri.randrange(3, 5) == rng.integers(3, 5)
    assert pri.choice([1, 2, 3]) == rng.choice([1, 2, 3])
    assert pri.gauss(0, 1) == rng.normal(0, 1)
    assert pri.expovariate(1.5) == rng.exponential(1 / 1.5)
    assert np.all(pri.shuffle([1, 2, 3]) == rng.shuffle([1, 2, 3]))
    assert np.all(pri.sample([1, 2, 3], 2) == rng.choice([1, 2, 3], (2,), replace=False))
    assert np.all([pri.randint(3, 5) for _ in range(100)] == [rng.integers(3, 6) for _ in range(100)])
    assert pri.random() == rng.random()
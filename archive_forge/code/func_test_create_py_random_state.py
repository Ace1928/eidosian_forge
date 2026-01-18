import random
from copy import copy
import pytest
import networkx as nx
from networkx.utils import (
from networkx.utils.misc import _dict_to_numpy_array1, _dict_to_numpy_array2
def test_create_py_random_state():
    pyrs = random.Random
    assert isinstance(create_py_random_state(1), pyrs)
    assert isinstance(create_py_random_state(None), pyrs)
    assert isinstance(create_py_random_state(pyrs(1)), pyrs)
    pytest.raises(ValueError, create_py_random_state, 'a')
    np = pytest.importorskip('numpy')
    rs = np.random.RandomState
    rng = np.random.default_rng(1000)
    rng_explicit = np.random.Generator(np.random.SFC64())
    nprs = PythonRandomInterface
    assert isinstance(create_py_random_state(np.random), nprs)
    assert isinstance(create_py_random_state(rs(1)), nprs)
    assert isinstance(create_py_random_state(rng), nprs)
    assert isinstance(create_py_random_state(rng_explicit), nprs)
    assert isinstance(PythonRandomInterface(), nprs)
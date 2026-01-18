import random
from copy import copy
import pytest
import networkx as nx
from networkx.utils import (
from networkx.utils.misc import _dict_to_numpy_array1, _dict_to_numpy_array2
@pytest.mark.parametrize(('iterable_type', 'expected'), ((list, 1), (tuple, 1), (str, '['), (set, 1)))
def test_arbitrary_element(iterable_type, expected):
    iterable = iterable_type([1, 2, 3])
    assert arbitrary_element(iterable) == expected
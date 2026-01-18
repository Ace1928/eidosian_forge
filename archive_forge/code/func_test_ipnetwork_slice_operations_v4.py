import pickle
import types
import random
import pytest
from netaddr import (
@pytest.mark.parametrize(['start', 'stop', 'step'], [[None, None, -1], [-1, 0, -2], [-1, None, -1], [-1, None, -3], [0, None, 4], [0, -1, None], [0, 4, None], [1, None, 4], [1, 4, 2]])
def test_ipnetwork_slice_operations_v4(start, stop, step):
    ip = IPNetwork('192.0.2.16/29')
    result = ip[start:stop:step]
    assert isinstance(result, types.GeneratorType)
    assert list(result) == list(ip)[start:stop:step]
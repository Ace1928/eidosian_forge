from typing import Dict, List
import catalogue
import pytest
from pytest import raises
from confection import Config, SimpleFrozenDict, SimpleFrozenList, registry
def test_frozen_dict():
    frozen = SimpleFrozenDict({k: k for k in range(10)})
    for k in range(10):
        assert frozen[k] == k
    with raises(NotImplementedError, match='frozen dictionary'):
        frozen[0] = 1
    with raises(NotImplementedError, match='frozen dictionary'):
        frozen[10] = 1
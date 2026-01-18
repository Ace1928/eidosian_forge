import os
import pytest
from monty.collections import AttrDict, FrozenAttrDict, Namespace, frozendict, tree
def test_namespace_dict(self):
    d = Namespace(foo='bar')
    d['hello'] = 'world'
    assert d['foo'] == 'bar'
    with pytest.raises(KeyError):
        d.update({'foo': 'spam'})
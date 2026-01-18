import numpy as np
import pytest
from ..volumeutils import DtypeMapper, Recoder, native_code, swapped_code
def test_custom_dicter():

    class MyDict:

        def __init__(self):
            self._keys = []

        def __setitem__(self, key, value):
            self._keys.append(key)

        def __getitem__(self, key):
            if key in self._keys:
                return 'spam'
            return 'eggs'

        def keys(self):
            return ['some', 'keys']

        def values(self):
            return ['funny', 'list']
    codes = ((1, 'one', '1', 'first'), (2, 'two'))
    rc = Recoder(codes, map_maker=MyDict)
    assert rc.code[1] == 'spam'
    assert rc.code['one'] == 'spam'
    assert rc.code['first'] == 'spam'
    assert rc.code['bizarre'] == 'eggs'
    assert rc.value_set() == {'funny', 'list'}
    assert list(rc.keys()) == ['some', 'keys']
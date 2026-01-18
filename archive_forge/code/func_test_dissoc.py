from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
def test_dissoc(self):
    D, kw = (self.D, self.kw)
    assert dissoc(D({'a': 1}), 'a', **kw) == D({})
    assert dissoc(D({'a': 1, 'b': 2}), 'a', **kw) == D({'b': 2})
    assert dissoc(D({'a': 1, 'b': 2}), 'b', **kw) == D({'a': 1})
    assert dissoc(D({'a': 1, 'b': 2}), 'a', 'b', **kw) == D({})
    assert dissoc(D({'a': 1}), 'a', **kw) == dissoc(dissoc(D({'a': 1}), 'a', **kw), 'a', **kw)
    d = D({'x': 1})
    oldd = d
    d2 = dissoc(d, 'x', **kw)
    assert d is oldd
    assert d2 is not oldd
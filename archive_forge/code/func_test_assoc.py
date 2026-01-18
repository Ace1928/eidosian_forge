from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
def test_assoc(self):
    D, kw = (self.D, self.kw)
    assert assoc(D({}), 'a', 1, **kw) == D({'a': 1})
    assert assoc(D({'a': 1}), 'a', 3, **kw) == D({'a': 3})
    assert assoc(D({'a': 1}), 'b', 3, **kw) == D({'a': 1, 'b': 3})
    d = D({'x': 1})
    oldd = d
    assoc(d, 'x', 2, **kw)
    assert d is oldd
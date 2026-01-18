from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
def test_assoc_in(self):
    D, kw = (self.D, self.kw)
    assert assoc_in(D({'a': 1}), ['a'], 2, **kw) == D({'a': 2})
    assert assoc_in(D({'a': D({'b': 1})}), ['a', 'b'], 2, **kw) == D({'a': D({'b': 2})})
    assert assoc_in(D({}), ['a', 'b'], 1, **kw) == D({'a': D({'b': 1})})
    d = D({'x': 1})
    oldd = d
    d2 = assoc_in(d, ['x'], 2, **kw)
    assert d is oldd
    assert d2 is not oldd
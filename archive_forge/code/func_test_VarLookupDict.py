import sys
import __future__
import inspect
import tokenize
import ast
import numbers
import six
from patsy import PatsyError
from patsy.util import PushbackAdapter, no_pickling, assert_no_pickling
from patsy.tokens import (pretty_untokenize, normalize_token_spacing,
from patsy.compat import call_and_wrap_exc
import patsy.builtins
def test_VarLookupDict():
    d1 = {'a': 1}
    d2 = {'a': 2, 'b': 3}
    ds = VarLookupDict([d1, d2])
    assert ds['a'] == 1
    assert ds['b'] == 3
    assert 'a' in ds
    assert 'c' not in ds
    import pytest
    pytest.raises(KeyError, ds.__getitem__, 'c')
    ds['a'] = 10
    assert ds['a'] == 10
    assert d1['a'] == 1
    assert ds.get('c') is None
    assert isinstance(repr(ds), six.string_types)
    assert_no_pickling(ds)
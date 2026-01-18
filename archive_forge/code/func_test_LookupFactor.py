import itertools
import numpy as np
from patsy import PatsyError
from patsy.categorical import C
from patsy.util import no_pickling, assert_no_pickling
def test_LookupFactor():
    l_a = LookupFactor('a')
    assert l_a.name() == 'a'
    assert l_a == LookupFactor('a')
    assert l_a != LookupFactor('b')
    assert hash(l_a) == hash(LookupFactor('a'))
    assert hash(l_a) != hash(LookupFactor('b'))
    assert l_a.eval({}, {'a': 1}) == 1
    assert l_a.eval({}, {'a': 2}) == 2
    assert repr(l_a) == "LookupFactor('a')"
    assert l_a.origin is None
    l_with_origin = LookupFactor('b', origin='asdf')
    assert l_with_origin.origin == 'asdf'
    l_c = LookupFactor('c', force_categorical=True, contrast='CONTRAST', levels=(1, 2))
    box = l_c.eval({}, {'c': [1, 1, 2]})
    assert box.data == [1, 1, 2]
    assert box.contrast == 'CONTRAST'
    assert box.levels == (1, 2)
    import pytest
    pytest.raises(ValueError, LookupFactor, 'nc', contrast='CONTRAST')
    pytest.raises(ValueError, LookupFactor, 'nc', levels=(1, 2))
    assert_no_pickling(LookupFactor('a'))
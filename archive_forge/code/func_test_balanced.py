import itertools
import numpy as np
from patsy import PatsyError
from patsy.categorical import C
from patsy.util import no_pickling, assert_no_pickling
def test_balanced():
    data = balanced(a=2, b=3)
    assert data['a'] == ['a1', 'a1', 'a1', 'a2', 'a2', 'a2']
    assert data['b'] == ['b1', 'b2', 'b3', 'b1', 'b2', 'b3']
    data = balanced(a=2, b=3, repeat=2)
    assert data['a'] == ['a1', 'a1', 'a1', 'a2', 'a2', 'a2', 'a1', 'a1', 'a1', 'a2', 'a2', 'a2']
    assert data['b'] == ['b1', 'b2', 'b3', 'b1', 'b2', 'b3', 'b1', 'b2', 'b3', 'b1', 'b2', 'b3']
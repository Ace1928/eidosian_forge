import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def test_case_sensitivity(self):
    """Test case sensitivity"""
    names = ['A', 'a', 'b', 'c']
    test = NameValidator().validate(names)
    assert_equal(test, ['A', 'a', 'b', 'c'])
    test = NameValidator(case_sensitive=False).validate(names)
    assert_equal(test, ['A', 'A_1', 'B', 'C'])
    test = NameValidator(case_sensitive='upper').validate(names)
    assert_equal(test, ['A', 'A_1', 'B', 'C'])
    test = NameValidator(case_sensitive='lower').validate(names)
    assert_equal(test, ['a', 'a_1', 'b', 'c'])
    assert_raises(ValueError, NameValidator, case_sensitive='foobar')
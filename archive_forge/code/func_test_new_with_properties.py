import os
import string
import pytest
from .util import random_string
from keyring import errors
def test_new_with_properties(self):
    alt = self.keyring.with_properties(foo='bar')
    assert alt is not self.keyring
    assert alt.foo == 'bar'
    with pytest.raises(AttributeError):
        self.keyring.foo
import itertools
import pytest
from referencing import Resource, exceptions
@pytest.mark.parametrize('one, two', pairs((each() for each in thunks)))
def test_eq_incompatible_types(one, two):
    assert one != two
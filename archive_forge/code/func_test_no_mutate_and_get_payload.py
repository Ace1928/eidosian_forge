from pytest import mark, raises
from ...types import (
from ...types.scalars import String
from ..mutation import ClientIDMutation
def test_no_mutate_and_get_payload():
    with raises(AssertionError) as excinfo:

        class MyMutation(ClientIDMutation):
            pass
    assert 'MyMutation.mutate_and_get_payload method is required in a ClientIDMutation.' == str(excinfo.value)
import inspect
import pytest
from ..utils import deprecation
from .utils import call_method
def test_deprecation_nested2():

    def level2():
        deprecation('test message', [])

    def level1():
        level2()
    with pytest.warns(DeprecationWarning) as record:
        call_method(level1)
    assert len(record) == 1
    assert record[0].filename == __file__
import numpy as np
import pytest
from pandas import Series
def test_is_unique_class_ne(capsys):

    class Foo:

        def __init__(self, val) -> None:
            self._value = val

        def __ne__(self, other):
            raise Exception('NEQ not supported')
    with capsys.disabled():
        li = [Foo(i) for i in range(5)]
        ser = Series(li, index=list(range(5)))
    ser.is_unique
    captured = capsys.readouterr()
    assert len(captured.err) == 0
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_metadata_propagation_indiv(self, monkeypatch):
    ser = Series(range(3), range(3))
    ser.name = 'foo'
    ser2 = Series(range(3), range(3))
    ser2.name = 'bar'
    result = ser.T
    tm.assert_metadata_equivalent(ser, result)

    def finalize(self, other, method=None, **kwargs):
        for name in self._metadata:
            if method == 'concat' and name == 'filename':
                value = '+'.join([getattr(obj, name) for obj in other.objs if getattr(obj, name, None)])
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, getattr(other, name, None))
        return self
    with monkeypatch.context() as m:
        m.setattr(Series, '_metadata', ['name', 'filename'])
        m.setattr(Series, '__finalize__', finalize)
        ser.filename = 'foo'
        ser2.filename = 'bar'
        result = pd.concat([ser, ser2])
        assert result.filename == 'foo+bar'
        assert result.name is None
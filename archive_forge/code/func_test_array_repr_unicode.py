import io
import pytest
import pandas as pd
def test_array_repr_unicode(self, data):
    result = str(data)
    assert isinstance(result, str)
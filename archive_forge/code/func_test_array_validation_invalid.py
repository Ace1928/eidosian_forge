import numpy
import pytest
from thinc.types import (
@pytest.mark.parametrize('arr,arr_type', [(numpy.zeros((0, 0), dtype=numpy.float32), Floats1d), (numpy.zeros((0, 0), dtype=numpy.float32), Ints2d)])
def test_array_validation_invalid(arr, arr_type):
    test_model = create_model('TestModel', arr=(arr_type, ...))
    with pytest.raises(ValidationError):
        test_model(arr=arr)
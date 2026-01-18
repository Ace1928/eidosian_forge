import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
@pytest.mark.parametrize('matched_axis', [0, 1])
@pytest.mark.parametrize('operator', ['__add__', '__sub__', '__truediv__', '__mul__', '__rtruediv__', '__rmul__', '__radd__', '__rsub__', '__ge__', '__gt__', '__lt__', '__le__', '__eq__', '__ne__'])
def test_binary_bad_broadcast(matched_axis, operator):
    """Tests broadcasts between 2d arrays that should fail."""
    if matched_axis == 0:
        operand1 = numpy.random.randint(-100, 100, size=(3, 100))
        operand2 = numpy.random.randint(-100, 100, size=(3, 200))
    else:
        operand1 = numpy.random.randint(-100, 100, size=(100, 3))
        operand2 = numpy.random.randint(-100, 100, size=(200, 3))
    with pytest.raises(ValueError):
        getattr(operand1, operator)(operand2)
    with pytest.raises(ValueError):
        getattr(np.array(operand1), operator)(np.array(operand2))
import pytest
import numpy as np
import cirq
@pytest.mark.parametrize('val,message', ((ReturnsNonnormalizedTuple(), '1.0'), (ReturnsNegativeProbability(), 'less than 0'), (ReturnsGreaterThanUnityProbability(), 'greater than 1')))
def test_invalid_mixture(val, message):
    with pytest.raises(ValueError, match=message):
        cirq.validate_mixture(val)
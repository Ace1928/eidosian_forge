import array
import pytest
import rpy2.rinterface as ri
def test_init_From_seq_invalid_float():
    seq = (1.0, 'b', 3.0)
    with pytest.raises(ValueError):
        ri.FloatSexpVector(seq)
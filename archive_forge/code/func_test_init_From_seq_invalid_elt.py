import pytest
import operator
from .. import utils
import rpy2.rinterface as ri
def test_init_From_seq_invalid_elt():
    seq = (ri.FloatSexpVector([1.0]), lambda x: x, ri.StrSexpVector(['foo', 'bar']))
    with pytest.raises(Exception):
        ri.ListSexpVector(seq)
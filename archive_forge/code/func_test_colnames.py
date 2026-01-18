import pytest
import rpy2.robjects as robjects
import array
def test_colnames():
    m = robjects.r.matrix(robjects.IntVector(range(4)), nrow=2, ncol=2)
    assert m.colnames == rinterface.NULL
    m.colnames = robjects.StrVector(('a', 'b'))
    assert len(m.colnames) == 2
    assert m.colnames[0] == 'a'
    assert m.colnames[1] == 'b'
    with pytest.raises(ValueError):
        m.colnames = robjects.StrVector(('a', 'b', 'c'))
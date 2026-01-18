import pytest
from statsmodels.tools.docstring import Docstring, remove_parameters, Parameter
def test_insert_parameters():
    new = Parameter('w', 'ndarray', ['An array input.'])
    ds = Docstring(good)
    ds.insert_parameters('y', new)
    assert 'w : ndarray' in str(ds)
    assert 'An array input.' in str(ds)
    other = Parameter('q', 'DataFrame', ['A pandas dataframe.'])
    ds = Docstring(good)
    ds.insert_parameters(None, [new, other])
    assert 'w : ndarray' in str(ds)
    assert 'An array input.' in str(ds)
    assert 'q : DataFrame' in str(ds)
    assert 'A pandas dataframe.' in str(ds)
    assert '---\nw : ndarray' in str(ds)
    ds = Docstring(good)
    with pytest.raises(ValueError):
        ds.insert_parameters('unknown', new)
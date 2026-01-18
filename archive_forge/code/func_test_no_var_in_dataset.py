import pytest
def test_no_var_in_dataset(ds):
    with pytest.raises(KeyError, match="No variable named 'foo'. Variables on the dataset include \\['z1', 'z2', 'x', 'time', 'c', 'y'\\]"):
        ds['foo']
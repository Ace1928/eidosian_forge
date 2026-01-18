from __future__ import annotations
import pytest
import dask
def test_no_mimesis():
    try:
        import mimesis
    except ImportError:
        with pytest.raises(Exception) as info:
            dask.datasets.make_people()
        assert 'python -m pip install mimesis' in str(info.value)
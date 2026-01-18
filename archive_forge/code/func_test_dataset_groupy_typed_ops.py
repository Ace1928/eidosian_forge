import numpy as np
from xarray import DataArray, Dataset, Variable
def test_dataset_groupy_typed_ops() -> None:
    """Tests for type checking of typed_ops on DatasetGroupBy"""
    ds = Dataset({'a': ('t', [1, 2, 3])}, coords={'x': ('t', [1, 2, 2])})
    grp = ds.groupby('x')

    def _test(ds: Dataset) -> None:
        assert isinstance(ds, Dataset)
    _da = DataArray([5, 6], coords={'x': [1, 2]}, dims='x')
    _ds = _da.to_dataset(name='a')
    _test(grp + _da)
    _test(grp + _ds)
    _test(_da + grp)
    _test(_ds + grp)
    _test(grp == _da)
    _test(_da == grp)
    _test(grp == _ds)
    _test(_ds == grp)
    _test(grp < _da)
    _test(_da > grp)
    _test(grp < _ds)
    _test(_ds > grp)
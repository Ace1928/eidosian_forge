from datashader import datashape
import pytest
def test_getitem_non_existent_typeset():
    with pytest.raises(KeyError):
        datashape.typesets.registry['footypeset']
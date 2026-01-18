import pytest
def test_namespace_nesting():
    with pytest.raises(ImportError):
        from networkx import networkx
import pytest
from geopandas._compat import import_optional_dependency
@pytest.mark.parametrize('bad_import', [['foo'], 0, False, True, {}, {'foo'}, {'foo': 'bar'}])
def test_import_optional_dependency_invalid(bad_import):
    with pytest.raises(ValueError, match='Invalid module name'):
        import_optional_dependency(bad_import)
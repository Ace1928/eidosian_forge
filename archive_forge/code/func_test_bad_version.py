import sys
import types
import pytest
from pandas.compat._optional import (
import pandas._testing as tm
def test_bad_version(monkeypatch):
    name = 'fakemodule'
    module = types.ModuleType(name)
    module.__version__ = '0.9.0'
    sys.modules[name] = module
    monkeypatch.setitem(VERSIONS, name, '1.0.0')
    match = "Pandas requires .*1.0.0.* of .fakemodule.*'0.9.0'"
    with pytest.raises(ImportError, match=match):
        import_optional_dependency('fakemodule')
    result = import_optional_dependency('fakemodule', min_version='0.8')
    assert result is module
    with tm.assert_produces_warning(UserWarning):
        result = import_optional_dependency('fakemodule', errors='warn')
    assert result is None
    module.__version__ = '1.0.0'
    result = import_optional_dependency('fakemodule')
    assert result is module
    with pytest.raises(ImportError, match="Pandas requires version '1.1.0'"):
        import_optional_dependency('fakemodule', min_version='1.1.0')
    with tm.assert_produces_warning(UserWarning):
        result = import_optional_dependency('fakemodule', errors='warn', min_version='1.1.0')
    assert result is None
    result = import_optional_dependency('fakemodule', errors='ignore', min_version='1.1.0')
    assert result is None
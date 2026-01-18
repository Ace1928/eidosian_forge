import sys
import types
import pytest
from pandas.compat._optional import (
import pandas._testing as tm
def test_import_optional():
    match = 'Missing .*notapackage.* pip .* conda .* notapackage'
    with pytest.raises(ImportError, match=match) as exc_info:
        import_optional_dependency('notapackage')
    assert isinstance(exc_info.value.__context__, ImportError)
    result = import_optional_dependency('notapackage', errors='ignore')
    assert result is None
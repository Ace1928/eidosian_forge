import sys
import pytest
def test_cannot_import_ssl(self):
    with pytest.raises(ImportError):
        import ssl
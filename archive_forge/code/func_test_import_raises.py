import pytest
from IPython.utils.importstring import import_item
def test_import_raises():
    """Test that failing imports raise the right exception"""
    pytest.raises(ImportError, import_item, 'IPython.foobar')
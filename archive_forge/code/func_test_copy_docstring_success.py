import datetime
import pytest  # type: ignore
from six.moves import urllib
from google.auth import _helpers
def test_copy_docstring_success():

    def func():
        pass
    _helpers.copy_docstring(SourceClass)(func)
    assert func.__doc__ == SourceClass.func.__doc__
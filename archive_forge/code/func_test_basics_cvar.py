import os
import sys
import pytest
from .. import (
def test_basics_cvar():
    with pytest.raises(AsyncLibraryNotFoundError):
        current_async_library()
    token = current_async_library_cvar.set('generic-lib')
    try:
        assert current_async_library() == 'generic-lib'
    finally:
        current_async_library_cvar.reset(token)
    with pytest.raises(AsyncLibraryNotFoundError):
        current_async_library()
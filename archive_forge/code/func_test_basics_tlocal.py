import os
import sys
import pytest
from .. import (
def test_basics_tlocal():
    with pytest.raises(AsyncLibraryNotFoundError):
        current_async_library()
    old_name, thread_local.name = (thread_local.name, 'generic-lib')
    try:
        assert current_async_library() == 'generic-lib'
    finally:
        thread_local.name = old_name
    with pytest.raises(AsyncLibraryNotFoundError):
        current_async_library()
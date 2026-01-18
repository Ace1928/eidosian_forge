from kivy import (
from unittest.mock import Mock, patch
from os.path import exists, isdir
def test_kivy_get_includes():
    """Test that the `get_includes` function return a list of valid paths."""
    paths = get_includes()
    assert len(paths) > 2, 'get_includes does not return a full path list.'
    for path in paths:
        assert exists(path) and isdir(path), 'get_includes returns invalid paths.'
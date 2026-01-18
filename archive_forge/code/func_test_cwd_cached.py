import os
import sys
import pytest
from nipype import config
from unittest.mock import MagicMock
def test_cwd_cached(tmpdir):
    """Check that changing dirs does not change nipype's cwd"""
    oldcwd = config.cwd
    tmpdir.chdir()
    assert config.cwd == oldcwd
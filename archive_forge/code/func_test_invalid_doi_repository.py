import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
def test_invalid_doi_repository():
    """Should fail if data repository is not supported"""
    with pytest.raises(ValueError) as exc:
        DOIDownloader()(url='doi:10.21105/joss.01943/file_name.txt', output_file=None, pooch=None)
    assert "Invalid data repository 'joss.theoj.org'" in str(exc.value)
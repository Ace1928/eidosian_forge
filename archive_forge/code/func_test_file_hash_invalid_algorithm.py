import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import pytest
from ..core import Pooch
from ..hashes import (
from .utils import check_tiny_data, mirror_directory
def test_file_hash_invalid_algorithm():
    """Test an invalid hashing algorithm"""
    with pytest.raises(ValueError) as exc:
        file_hash(fname='something', alg='blah')
    assert "'blah'" in str(exc.value)
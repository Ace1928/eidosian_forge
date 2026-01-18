import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import pytest
from ..core import Pooch
from ..hashes import (
from .utils import check_tiny_data, mirror_directory
@pytest.mark.parametrize('alg,expected_hash', list(TINY_DATA_HASHES_HASHLIB.items()), ids=list(TINY_DATA_HASHES_HASHLIB.keys()))
def test_hash_matches_strict(alg, expected_hash):
    """Make sure the hash checking function raises an exception if strict"""
    fname = os.path.join(DATA_DIR, 'tiny-data.txt')
    check_tiny_data(fname)
    known_hash = f'{alg}:{expected_hash}'
    assert hash_matches(fname, known_hash, strict=True)
    bad_hash = f'{alg}:blablablabla'
    with pytest.raises(ValueError) as error:
        hash_matches(fname, bad_hash, strict=True, source='Neverland')
    assert 'Neverland' in str(error.value)
    with pytest.raises(ValueError) as error:
        hash_matches(fname, bad_hash, strict=True, source=None)
    assert fname in str(error.value)
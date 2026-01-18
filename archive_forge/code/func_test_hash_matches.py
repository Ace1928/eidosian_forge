import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import pytest
from ..core import Pooch
from ..hashes import (
from .utils import check_tiny_data, mirror_directory
@pytest.mark.parametrize('alg,expected_hash', list(TINY_DATA_HASHES.items()), ids=list(TINY_DATA_HASHES.keys()))
def test_hash_matches(alg, expected_hash):
    """Make sure the hash checking function works"""
    if alg.startswith('xxh'):
        if xxhash is None:
            pytest.skip('requires xxhash')
        if alg not in ['xxh64', 'xxh32'] and XXHASH_MAJOR_VERSION < 2:
            pytest.skip('requires xxhash > 2.0')
    fname = os.path.join(DATA_DIR, 'tiny-data.txt')
    check_tiny_data(fname)
    known_hash = f'{alg}:{expected_hash}'
    assert hash_matches(fname, known_hash)
    known_hash = f'{alg}:blablablabla'
    assert not hash_matches(fname, known_hash)
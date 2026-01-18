import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import pytest
from ..core import Pooch
from ..hashes import (
from .utils import check_tiny_data, mirror_directory
def test_make_registry_recursive(data_dir_mirror):
    """Check that the registry builder works in recursive mode"""
    outfile = NamedTemporaryFile(delete=False)
    outfile.close()
    try:
        make_registry(data_dir_mirror, outfile.name, recursive=True)
        with open(outfile.name, encoding='utf-8') as fout:
            registry = fout.read()
        assert registry == REGISTRY_RECURSIVE
        pup = Pooch(path=data_dir_mirror, base_url='some bogus URL', registry={})
        pup.load_registry(outfile.name)
        assert str(data_dir_mirror / 'tiny-data.txt') == pup.fetch('tiny-data.txt')
        check_tiny_data(pup.fetch('tiny-data.txt'))
        true = str(data_dir_mirror / 'subdir' / 'tiny-data.txt')
        assert true == pup.fetch('subdir/tiny-data.txt')
        check_tiny_data(pup.fetch('subdir/tiny-data.txt'))
    finally:
        os.remove(outfile.name)
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from ..core import create, Pooch, retrieve, download_action, stream_download
from ..utils import get_logger, temporary_file, os_cache
from ..hashes import file_hash, hash_matches
from .. import core
from ..downloaders import HTTPDownloader, FTPDownloader
from .utils import (
def test_load_registry_from_doi_zenodo_with_slash():
    """
    Check that the registry is correctly populated from the Zenodo API when
    the filename contains a slash
    """
    url = ZENODOURL_W_SLASH
    with TemporaryDirectory() as local_store:
        path = os.path.abspath(local_store)
        pup = Pooch(path=path, base_url=url)
        pup.load_registry_from_doi()
        assert len(pup.registry) == 1
        assert 'santisoler/pooch-test-data-v1.zip' in pup.registry
        for filename in pup.registry:
            pup.fetch(filename)
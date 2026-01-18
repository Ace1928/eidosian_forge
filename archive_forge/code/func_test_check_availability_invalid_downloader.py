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
def test_check_availability_invalid_downloader():
    """Should raise an exception if the downloader doesn't support this"""

    def downloader(url, output, pooch):
        """A downloader that doesn't support check_only"""
        return None
    pup = Pooch(path=DATA_DIR, base_url=BASEURL, registry=REGISTRY)
    assert pup.is_available('tiny-data.txt')
    with pytest.raises(NotImplementedError) as error:
        pup.is_available('tiny-data.txt', downloader=downloader)
        assert 'does not support availability checks' in str(error)
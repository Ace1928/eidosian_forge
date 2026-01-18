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
@pytest.mark.network
def test_fetch_with_downloader(capsys):
    """Setup a downloader function for fetch"""

    def download(url, output_file, pup):
        """Download through HTTP and warn that we're doing it"""
        get_logger().info('downloader executed')
        HTTPDownloader()(url, output_file, pup)
    with TemporaryDirectory() as local_store:
        path = Path(local_store)
        pup = Pooch(path=path, base_url=BASEURL, registry=REGISTRY)
        with capture_log() as log_file:
            fname = pup.fetch('large-data.txt', downloader=download)
            logs = log_file.getvalue()
            lines = logs.splitlines()
            assert len(lines) == 2
            assert lines[0].split()[0] == 'Downloading'
            assert lines[1] == 'downloader executed'
        assert not capsys.readouterr().err
        check_large_data(fname)
        with capture_log() as log_file:
            fname = pup.fetch('large-data.txt')
            assert log_file.getvalue() == ''
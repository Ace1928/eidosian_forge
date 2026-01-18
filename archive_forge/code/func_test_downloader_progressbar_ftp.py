import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.network
@pytest.mark.skipif(tqdm is None, reason='requires tqdm')
def test_downloader_progressbar_ftp(capsys, ftpserver):
    """Setup an FTP downloader function that prints a progress bar for fetch"""
    with data_over_ftp(ftpserver, 'tiny-data.txt') as url:
        download = FTPDownloader(progressbar=True, port=ftpserver.server_port)
        with TemporaryDirectory() as local_store:
            outfile = os.path.join(local_store, 'tiny-data.txt')
            download(url, outfile, None)
            captured = capsys.readouterr()
            printed = captured.err.split('\r')[-1].strip()
            assert len(printed) == 79
            if sys.platform == 'win32':
                progress = '100%|####################'
            else:
                progress = '100%|████████████████████'
            assert printed[:25] == progress
            check_tiny_data(outfile)
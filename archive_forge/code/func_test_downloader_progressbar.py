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
@pytest.mark.parametrize('url,downloader', [(BASEURL, HTTPDownloader), (FIGSHAREURL, DOIDownloader)], ids=['http', 'figshare'])
def test_downloader_progressbar(url, downloader, capsys):
    """Setup a downloader function that prints a progress bar for fetch"""
    download = downloader(progressbar=True)
    with TemporaryDirectory() as local_store:
        fname = 'tiny-data.txt'
        url = url + fname
        outfile = os.path.join(local_store, fname)
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
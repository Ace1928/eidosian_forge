import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.parametrize('url', [FIGSHAREURL, ZENODOURL, DATAVERSEURL], ids=['figshare', 'zenodo', 'dataverse'])
def test_doi_downloader(url):
    """Test the DOI downloader"""
    with TemporaryDirectory() as local_store:
        downloader = DOIDownloader()
        outfile = os.path.join(local_store, 'tiny-data.txt')
        downloader(url + 'tiny-data.txt', outfile, None)
        check_tiny_data(outfile)
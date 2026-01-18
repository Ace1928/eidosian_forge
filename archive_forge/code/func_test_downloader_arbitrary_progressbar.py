import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.network
def test_downloader_arbitrary_progressbar(capsys):
    """Setup a downloader function with an arbitrary progress bar class."""

    class MinimalProgressDisplay:
        """A minimalist replacement for tqdm.tqdm"""

        def __init__(self, total):
            self.count = 0
            self.total = total

        def __repr__(self):
            """represent current completion"""
            return str(self.count) + '/' + str(self.total)

        def render(self):
            """print self.__repr__ to stderr"""
            print(f'\r{self}', file=sys.stderr, end='')

        def update(self, i):
            """modify completion and render"""
            self.count = i
            self.render()

        def reset(self):
            """set counter to 0"""
            self.count = 0

        @staticmethod
        def close():
            """print a new empty line"""
            print('', file=sys.stderr)
    pbar = MinimalProgressDisplay(total=None)
    download = HTTPDownloader(progressbar=pbar)
    with TemporaryDirectory() as local_store:
        fname = 'large-data.txt'
        url = BASEURL + fname
        outfile = os.path.join(local_store, 'large-data.txt')
        download(url, outfile, None)
        captured = capsys.readouterr()
        printed = captured.err.split('\r')[-1].strip()
        progress = '336/336'
        assert printed == progress
        check_large_data(outfile)
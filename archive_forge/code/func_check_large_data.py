import os
import io
import logging
import shutil
import stat
from pathlib import Path
from contextlib import contextmanager
from .. import __version__ as full_version
from ..utils import check_version, get_logger
def check_large_data(fname):
    """
    Load the large-data.txt file and check that the contents are correct.
    """
    assert os.path.exists(fname)
    with open(fname, encoding='utf-8') as data:
        content = data.read()
    true_content = ['# A larer data file for test purposes only']
    true_content.extend(['1  2  3  4  5  6'] * 6002)
    assert content.strip() == '\n'.join(true_content)
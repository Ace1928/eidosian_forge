import os
import io
import logging
import shutil
import stat
from pathlib import Path
from contextlib import contextmanager
from .. import __version__ as full_version
from ..utils import check_version, get_logger
def pooch_test_url():
    """
    Get the base URL for the test data used in Pooch itself.

    The URL is a GitHub raw link to the ``pooch/tests/data`` directory from the
    `GitHub repository <https://github.com/fatiando/pooch>`__. It matches the
    pooch version specified in ``pooch.version.full_version``.

    Returns
    -------
    url
        The versioned URL for pooch's test data.

    """
    version = check_version(full_version, fallback='main')
    url = f'https://github.com/fatiando/pooch/raw/{version}/pooch/tests/data/'
    return url
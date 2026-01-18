import os
import io
import logging
import shutil
import stat
from pathlib import Path
from contextlib import contextmanager
from .. import __version__ as full_version
from ..utils import check_version, get_logger
def pooch_test_zenodo_url():
    """
    Get the base URL for the test data stored in Zenodo.

    The URL contains the DOI for the Zenodo dataset using the appropriate
    version for this version of Pooch.

    Returns
    -------
    url
        The URL for pooch's test data.

    """
    url = 'doi:10.5281/zenodo.4924875/'
    return url
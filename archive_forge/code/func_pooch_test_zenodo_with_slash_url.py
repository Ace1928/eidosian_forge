import os
import io
import logging
import shutil
import stat
from pathlib import Path
from contextlib import contextmanager
from .. import __version__ as full_version
from ..utils import check_version, get_logger
def pooch_test_zenodo_with_slash_url():
    """
    Get base URL for test data in Zenodo, where the file name contains a slash

    The URL contains the DOI for the Zenodo dataset that has a slash in the
    filename (created with the GitHub-Zenodo integration service), using the
    appropriate version for this version of Pooch.

    Returns
    -------
    url
        The URL for pooch's test data.

    """
    url = 'doi:10.5281/zenodo.7632643/'
    return url
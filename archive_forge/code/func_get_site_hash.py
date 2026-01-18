import hashlib
import os
import site
import sys
import tarfile
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional
from wasabi import msg
from ..errors import Errors
from ..util import check_spacy_env_vars, download_file, ensure_pathy, get_checksum
from ..util import get_hash, make_tempdir, upload_file
def get_site_hash():
    """Hash the current Python environment's site-packages contents, including
    the name and version of the libraries. The list we're hashing is what
    `pip freeze` would output.
    """
    site_dirs = site.getsitepackages()
    if site.ENABLE_USER_SITE:
        site_dirs.extend(site.getusersitepackages())
    packages = set()
    for site_dir in site_dirs:
        site_dir = Path(site_dir)
        for subpath in site_dir.iterdir():
            if subpath.parts[-1].endswith('dist-info'):
                packages.add(subpath.parts[-1].replace('.dist-info', ''))
    package_bytes = ''.join(sorted(packages)).encode('utf8')
    return hashlib.md5sum(package_bytes).hexdigest()
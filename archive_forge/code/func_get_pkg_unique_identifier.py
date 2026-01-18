from __future__ import annotations
import errno
import hashlib
import json
import logging
import os
import sys
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import (
import requests
from filelock import FileLock
def get_pkg_unique_identifier() -> str:
    """Generate an identifier unique to the python version, tldextract version, and python instance.

    This will prevent interference between virtualenvs and issues that might arise when installing
    a new version of tldextract
    """
    try:
        from tldextract._version import version
    except ImportError:
        version = 'dev'
    tldextract_version = 'tldextract-' + version
    python_env_name = os.path.basename(sys.prefix)
    python_binary_path_short_hash = md5(sys.prefix.encode('utf-8')).hexdigest()[:6]
    python_version = '.'.join([str(v) for v in sys.version_info[:-1]])
    identifier_parts = [python_version, python_env_name, python_binary_path_short_hash, tldextract_version]
    pkg_identifier = '__'.join(identifier_parts)
    return pkg_identifier
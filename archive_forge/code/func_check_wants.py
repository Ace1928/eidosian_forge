import logging
import os
import select
import socket
import subprocess
import sys
from contextlib import closing
from io import BufferedReader, BytesIO
from typing import (
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urljoin, urlparse, urlunparse, urlunsplit
import dulwich
from .config import Config, apply_instead_of, get_xdg_config_home_path
from .errors import GitProtocolError, NotGitRepository, SendPackError
from .pack import (
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, _import_remote_refs, read_info_refs
from .repo import Repo
def check_wants(wants, refs):
    """Check that a set of wants is valid.

    Args:
      wants: Set of object SHAs to fetch
      refs: Refs dictionary to check against
    """
    missing = set(wants) - {v for k, v in refs.items() if not k.endswith(PEELED_TAG_SUFFIX)}
    if missing:
        raise InvalidWants(missing)
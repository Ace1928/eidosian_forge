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
def get_credentials_from_store(scheme, hostname, username=None, fnames=DEFAULT_GIT_CREDENTIALS_PATHS):
    for fname in fnames:
        try:
            with open(fname, 'rb') as f:
                for line in f:
                    parsed_line = urlparse(line.strip())
                    if parsed_line.scheme == scheme and parsed_line.hostname == hostname and (username is None or parsed_line.username == username):
                        return (parsed_line.username, parsed_line.password)
        except FileNotFoundError:
            continue
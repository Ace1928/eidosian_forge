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
def read_pkt_refs(pkt_seq):
    server_capabilities = None
    refs = {}
    for pkt in pkt_seq:
        sha, ref = pkt.rstrip(b'\n').split(None, 1)
        if sha == b'ERR':
            raise GitProtocolError(ref.decode('utf-8', 'replace'))
        if server_capabilities is None:
            ref, server_capabilities = extract_capabilities(ref)
        refs[ref] = sha
    if len(refs) == 0:
        return ({}, set())
    if refs == {CAPABILITIES_REF: ZERO_SHA}:
        refs = {}
    return (refs, set(server_capabilities))
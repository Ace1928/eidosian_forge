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
def body_generator():
    header_handler = _v1ReceivePackHeader(negotiated_capabilities, old_refs, new_refs)
    for pkt in header_handler:
        yield pkt_line(pkt)
    pack_data_count, pack_data = generate_pack_data(header_handler.have, header_handler.want, ofs_delta=CAPABILITY_OFS_DELTA in negotiated_capabilities)
    if self._should_send_pack(new_refs):
        yield from PackChunkGenerator(pack_data_count, pack_data)
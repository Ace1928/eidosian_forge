import collections
import os
import socket
import sys
import time
from functools import partial
from typing import Dict, Iterable, List, Optional, Set, Tuple
import socketserver
import zlib
from dulwich import log_utils
from .archive import tar_stream
from .errors import (
from .object_store import peel_sha
from .objects import Commit, ObjectID, valid_hexsha
from .pack import ObjectContainer, PackedObjectContainer, write_pack_from_container
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, RefsContainer, write_info_refs
from .repo import BaseRepo, Repo
def set_client_capabilities(self, caps: Iterable[bytes]) -> None:
    allowable_caps = set(self.innocuous_capabilities())
    allowable_caps.update(self.capabilities())
    for cap in caps:
        if cap.startswith(CAPABILITY_AGENT + b'='):
            continue
        if cap not in allowable_caps:
            raise GitProtocolError('Client asked for capability %r that was not advertised.' % cap)
    for cap in self.required_capabilities():
        if cap not in caps:
            raise GitProtocolError('Client does not support required capability %r.' % cap)
    self._client_capabilities = set(caps)
    logger.info('Client capabilities: %s', caps)
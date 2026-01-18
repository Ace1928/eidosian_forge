import bz2
import itertools
import os
import queue
import sys
import tempfile
import threading
import zlib
import fastbencode as bencode
from ... import errors, estimate_compressed_size, osutils
from ... import revision as _mod_revision
from ... import trace, ui
from ...repository import _strip_NULL_ghosts, network_format_registry
from .. import inventory as _mod_inventory
from .. import inventory_delta, pack, vf_search
from ..bzrdir import BzrDir
from ..versionedfile import (ChunkedContentFactory, NetworkRecordStream,
from .request import (FailedSmartServerResponse, SmartServerRequest,
def do_body(self, body_bytes):
    repository = self._repository
    repository.lock_read()
    try:
        source = repository._get_source(self._to_format)
        keys = []
        for entry in body_bytes.split(b'\n'):
            kind, revid = entry.split(b'\t')
            keys.append((kind.decode('utf-8'), revid))
        stream = source.get_stream_for_missing_keys(keys)
    except Exception:
        try:
            repository.unlock()
        finally:
            raise
    return SuccessfulSmartServerResponse((b'ok',), body_stream=self.body_stream(stream, repository))
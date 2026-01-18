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
def iter_stream_decoder(self):
    """Iterate the contents of the pack from stream_decoder."""
    yield from self.stream_decoder.read_pending_records()
    for bytes in self.byte_stream:
        self.stream_decoder.accept_bytes(bytes)
        yield from self.stream_decoder.read_pending_records()
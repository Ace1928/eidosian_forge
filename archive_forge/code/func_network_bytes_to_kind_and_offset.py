import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
def network_bytes_to_kind_and_offset(network_bytes):
    """Strip of a record kind from the front of network_bytes.

    :param network_bytes: The bytes of a record.
    :return: A tuple (storage_kind, offset_of_remaining_bytes)
    """
    line_end = network_bytes.find(b'\n')
    storage_kind = network_bytes[:line_end].decode('ascii')
    return (storage_kind, line_end + 1)
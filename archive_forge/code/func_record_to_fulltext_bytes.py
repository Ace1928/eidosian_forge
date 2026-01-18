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
def record_to_fulltext_bytes(record):
    if record.parents is None:
        parents = b'nil'
    else:
        parents = tuple([tuple(p) for p in record.parents])
    record_meta = bencode.bencode((record.key, parents))
    record_content = record.get_bytes_as('fulltext')
    return b'fulltext\n%s%s%s' % (_length_prefix(record_meta), record_meta, record_content)
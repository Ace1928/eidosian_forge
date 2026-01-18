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
def add_lines_with_ghosts(self, version_id, parents, lines, parent_texts=None, nostore_sha=None, random_id=False, check_content=True, left_matching_blocks=None):
    """Add lines to the versioned file, allowing ghosts to be present.

        This takes the same parameters as add_lines and returns the same.
        """
    self._check_write_ok()
    return self._add_lines_with_ghosts(version_id, parents, lines, parent_texts, nostore_sha, random_id, check_content, left_matching_blocks)
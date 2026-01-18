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
def base_from_plan(self):
    """Construct a BASE file from the plan text."""
    base_lines = []
    for state, line in self.plan:
        if state in ('killed-a', 'killed-b', 'killed-both', 'unchanged'):
            base_lines.append(line)
        elif state not in ('killed-base', 'irrelevant', 'ghost-a', 'ghost-b', 'new-a', 'new-b', 'conflicted-a', 'conflicted-b'):
            raise AssertionError('Unknown state: {}'.format(state))
    return base_lines
from io import BytesIO
import os
import os.path as osp
from pathlib import Path
from stat import (
import subprocess
from git.cmd import handle_process_output, safer_popen
from git.compat import defenc, force_bytes, force_text, safe_decode
from git.exc import HookExecutionError, UnmergedEntriesError
from git.objects.fun import (
from git.util import IndexFileSHA1Writer, finalize_process
from gitdb.base import IStream
from gitdb.typ import str_tree_type
from .typ import BaseIndexEntry, IndexEntry, CE_NAMEMASK, CE_STAGESHIFT
from .util import pack, unpack
from typing import Dict, IO, List, Sequence, TYPE_CHECKING, Tuple, Type, Union, cast
from git.types import PathLike
def read_header(stream: IO[bytes]) -> Tuple[int, int]:
    """Return tuple(version_long, num_entries) from the given stream"""
    type_id = stream.read(4)
    if type_id != b'DIRC':
        raise AssertionError('Invalid index file header: %r' % type_id)
    unpacked = cast(Tuple[int, int], unpack('>LL', stream.read(4 * 2)))
    version, num_entries = unpacked
    assert version in (1, 2)
    return (version, num_entries)
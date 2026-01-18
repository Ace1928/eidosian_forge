import contextlib
import errno
import itertools
import os
from io import BytesIO
from stat import S_IFDIR, S_IFLNK, S_IFREG, S_ISDIR
from .. import transport, urlutils
from ..errors import InProcessTransport, LockError, TransportNotPossible
from ..transport import (AppendBasedFileStream, FileExists, LateReadError,
def memory_factory(url):
    from . import memory
    result = memory.MemoryTransport(url)
    result._dirs = self._dirs
    result._files = self._files
    result._symlinks = self._symlinks
    result._locks = self._locks
    return result
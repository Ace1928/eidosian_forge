import contextlib
import errno
import itertools
import os
from io import BytesIO
from stat import S_IFDIR, S_IFLNK, S_IFREG, S_ISDIR
from .. import transport, urlutils
from ..errors import InProcessTransport, LockError, TransportNotPossible
from ..transport import (AppendBasedFileStream, FileExists, LateReadError,
def do_renames(container):
    renames = []
    for path in container:
        new_path = replace(path)
        if new_path != path:
            if new_path in container:
                raise FileExists(new_path)
            renames.append((path, new_path))
    for path, new_path in renames:
        container[new_path] = container[path]
        del container[path]
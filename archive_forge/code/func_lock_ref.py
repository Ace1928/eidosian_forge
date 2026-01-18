import os
import posixpath
import sys
from io import BytesIO
from dulwich.errors import NoIndexPresent
from dulwich.file import FileLocked, _GitFile
from dulwich.object_store import (PACK_MODE, PACKDIR, PackBasedObjectStore,
from dulwich.objects import ShaFile
from dulwich.pack import (PACK_SPOOL_FILE_MAX_SIZE, MemoryPackIndex, Pack,
from dulwich.refs import SymrefLoop
from dulwich.repo import (BASE_DIRECTORIES, COMMONDIR, CONTROLDIR,
from .. import osutils
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..errors import (AlreadyControlDirError, LockBroken, LockContention,
from ..lock import LogicalLockResult
from ..trace import warning
from ..transport import FileExists, NoSuchFile
from ..transport.local import LocalTransport
def lock_ref(self, name):
    if name == b'HEAD':
        transport = self.worktree_transport
    else:
        transport = self.transport
    self._ensure_dir_exists(urlutils.quote_from_bytes(name))
    lockname = urlutils.quote_from_bytes(name + b'.lock')
    try:
        local_path = transport.local_abspath(urlutils.quote_from_bytes(name))
    except NotLocalUrl:
        if transport.has(lockname):
            raise LockContention(name)
        transport.put_bytes(lockname, b'Locked by brz-git')
        return LogicalLockResult(lambda: transport.delete(lockname))
    else:
        try:
            gf = TransportGitFile(transport, urlutils.quote_from_bytes(name), 'wb')
        except FileLocked as e:
            raise LockContention(name, e)
        else:

            def unlock():
                try:
                    transport.delete(lockname)
                except NoSuchFile:
                    raise LockBroken(lockname)
                gf.abort()
            return LogicalLockResult(unlock)
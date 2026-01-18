import os
import time
import yaml
from . import config, debug, errors, lock, osutils, ui, urlutils
from .decorators import only_raises
from .errors import (DirectoryNotEmpty, LockBreakMismatch, LockBroken,
from .i18n import gettext
from .osutils import format_delta, get_host_name, rand_chars
from .trace import mutter, note
from .transport import FileExists, NoSuchFile
def force_break(self, dead_holder_info):
    """Release a lock held by another process.

        WARNING: This should only be used when the other process is dead; if
        it still thinks it has the lock there will be two concurrent writers.
        In general the user's approval should be sought for lock breaks.

        After the lock is broken it will not be held by any process.
        It is possible that another process may sneak in and take the
        lock before the breaking process acquires it.

        :param dead_holder_info:
            Must be the result of a previous LockDir.peek() call; this is used
            to check that it's still held by the same process that the user
            decided was dead.  If this is not the current holder,
            LockBreakMismatch is raised.

        :returns: LockResult for the broken lock.
        """
    if not isinstance(dead_holder_info, LockHeldInfo):
        raise ValueError('dead_holder_info: %r' % dead_holder_info)
    self._check_not_locked()
    current_info = self.peek()
    if current_info is None:
        return
    if current_info != dead_holder_info:
        raise LockBreakMismatch(self, current_info, dead_holder_info)
    tmpname = '{}/broken.{}.tmp'.format(self.path, rand_chars(20))
    self.transport.rename(self._held_dir, tmpname)
    broken_info_path = tmpname + self.__INFO_NAME
    broken_info = self._read_info_file(broken_info_path)
    if broken_info != dead_holder_info:
        raise LockBreakMismatch(self, broken_info, dead_holder_info)
    self.transport.delete(broken_info_path)
    self.transport.rmdir(tmpname)
    result = lock.LockResult(self.transport.abspath(self.path), current_info.nonce)
    for hook in self.hooks['lock_broken']:
        hook(result)
    return result
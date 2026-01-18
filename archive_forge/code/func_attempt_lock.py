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
def attempt_lock(self):
    """Take the lock; fail if it's already held.

        If you wish to block until the lock can be obtained, call wait_lock()
        instead.

        :return: The lock token.
        :raises LockContention: if the lock is held by someone else.
        """
    if self._fake_read_lock:
        raise LockContention(self)
    result = self._attempt_lock()
    hook_result = lock.LockResult(self.transport.abspath(self.path), self.nonce)
    for hook in self.hooks['lock_acquired']:
        hook(hook_result)
    return result
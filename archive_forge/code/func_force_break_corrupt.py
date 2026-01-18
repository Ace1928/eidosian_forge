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
def force_break_corrupt(self, corrupt_info_lines):
    """Release a lock that has been corrupted.

        This is very similar to force_break, it except it doesn't assume that
        self.peek() can work.

        :param corrupt_info_lines: the lines of the corrupted info file, used
            to check that the lock hasn't changed between reading the (corrupt)
            info file and calling force_break_corrupt.
        """
    self._check_not_locked()
    tmpname = '{}/broken.{}.tmp'.format(self.path, rand_chars(20))
    self.transport.rename(self._held_dir, tmpname)
    broken_info_path = tmpname + self.__INFO_NAME
    broken_content = self.transport.get_bytes(broken_info_path)
    broken_lines = osutils.split_lines(broken_content)
    if broken_lines != corrupt_info_lines:
        raise LockBreakMismatch(self, broken_lines, corrupt_info_lines)
    self.transport.delete(broken_info_path)
    self.transport.rmdir(tmpname)
    result = lock.LockResult(self.transport.abspath(self.path))
    for hook in self.hooks['lock_broken']:
        hook(result)
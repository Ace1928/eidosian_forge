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
def is_locked_by_this_process(self):
    """True if this process seems to be the current lock holder."""
    return self.get('hostname') == get_host_name() and self.get('pid') == os.getpid() and (self.get('user') == get_username_for_lock_info())
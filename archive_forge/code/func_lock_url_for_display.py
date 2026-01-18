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
def lock_url_for_display(self):
    """Give a nicely-printable representation of the URL of this lock."""
    lock_url = self.transport.abspath(self.path)
    if lock_url.startswith('file://'):
        lock_url = lock_url.split('.bzr/')[0]
    else:
        lock_url = ''
    return lock_url
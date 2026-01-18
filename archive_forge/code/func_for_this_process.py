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
@classmethod
def for_this_process(cls, extra_holder_info):
    """Return a new LockHeldInfo for a lock taken by this process.
        """
    info = dict(hostname=get_host_name(), pid=os.getpid(), nonce=rand_chars(20), start_time=int(time.time()), user=get_username_for_lock_info())
    if extra_holder_info is not None:
        info.update(extra_holder_info)
    return cls(info)
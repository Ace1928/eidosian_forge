from __future__ import with_statement
from wandb_watchdog.utils import platform
import threading
import errno
import sys
import stat
import os
from wandb_watchdog.observers.api import (
from wandb_watchdog.utils.dirsnapshot import DirectorySnapshot
from wandb_watchdog.events import (
def is_renamed(kev):
    """Determines whether the given kevent represents movement."""
    return kev.fflags & select.KQ_NOTE_RENAME
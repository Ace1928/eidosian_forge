import asyncio
import asyncio.exceptions
import atexit
import errno
import os
import signal
import sys
import time
from subprocess import CalledProcessError
from threading import Thread
from traitlets import Any, Dict, List, default
from IPython.core import magic_arguments
from IPython.core.async_helpers import _AsyncIOProxy
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from IPython.utils.process import arg_split
def kill_bg_processes(self):
    """Kill all BG processes which are still running."""
    if not self.bg_processes:
        return
    for p in self.bg_processes:
        if p.returncode is None:
            try:
                p.send_signal(signal.SIGINT)
            except:
                pass
    time.sleep(0.1)
    self._gc_bg_processes()
    if not self.bg_processes:
        return
    for p in self.bg_processes:
        if p.returncode is None:
            try:
                p.terminate()
            except:
                pass
    time.sleep(0.1)
    self._gc_bg_processes()
    if not self.bg_processes:
        return
    for p in self.bg_processes:
        if p.returncode is None:
            try:
                p.kill()
            except:
                pass
    self._gc_bg_processes()
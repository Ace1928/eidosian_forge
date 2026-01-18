from __future__ import annotations
import os
import select
import shlex
import signal
import subprocess
import sys
from typing import ClassVar, Mapping
import param
from pyviz_comms import JupyterComm
from ..io.callbacks import PeriodicCallback
from ..util import edit_readonly, lazy_load
from .base import Widget
@property
def subprocess(self):
    """
        The subprocess enables running commands like 'ls', ['ls',
        '-l'], 'bash', 'python' and 'ipython' in the terminal.
        """
    if not self._subprocess:
        self._subprocess = TerminalSubprocess(self)
    return self._subprocess
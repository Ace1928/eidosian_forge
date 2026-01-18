from __future__ import annotations
import os
import shlex
import sys
import typing as t
from shutil import which
from jupyter_core.utils import ensure_async
from jupyter_server.extension.application import ExtensionApp
from jupyter_server.transutils import trans
from traitlets import Type
from . import api_handlers, handlers
from .terminalmanager import TerminalManager
def initialize_configurables(self) -> None:
    """Initialize configurables."""
    default_shell = 'powershell.exe' if os.name == 'nt' else which('sh')
    assert self.serverapp is not None
    shell_override = self.serverapp.terminado_settings.get('shell_command')
    if isinstance(shell_override, str):
        shell_override = shlex.split(shell_override)
    shell = [os.environ.get('SHELL') or default_shell] if shell_override is None else shell_override
    if os.name != 'nt' and shell_override is None and (not sys.stdout.isatty()):
        shell.append('-l')
    self.terminal_manager = self.terminal_manager_class(shell_command=shell, extra_env={'JUPYTER_SERVER_ROOT': self.serverapp.root_dir, 'JUPYTER_SERVER_URL': self.serverapp.connection_url}, parent=self.serverapp)
    self.terminal_manager.log = self.serverapp.log
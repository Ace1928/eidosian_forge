from __future__ import annotations
import abc
import shlex
import tempfile
import typing as t
from .io import (
from .config import (
from .util import (
from .util_common import (
from .docker_util import (
from .ssh import (
from .become import (
@staticmethod
def capture_log_details(path: str, ex: SubprocessError) -> None:
    """Read the specified SSH debug log and add relevant details to the provided exception."""
    if ex.status != 255:
        return
    markers = ['debug1: Connection Established', 'debug1: Authentication successful', 'debug1: Entering interactive session', 'debug1: Sending command', 'debug2: PTY allocation request accepted', 'debug2: exec request accepted']
    file_contents = read_text_file(path)
    messages = []
    for line in reversed(file_contents.splitlines()):
        messages.append(line)
        if any((line.startswith(marker) for marker in markers)):
            break
    message = '\n'.join(reversed(messages))
    ex.message += '>>> SSH Debug Output\n'
    ex.message += '%s%s\n' % (message.strip(), Display.clear)
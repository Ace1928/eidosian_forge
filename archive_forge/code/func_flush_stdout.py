from __future__ import annotations
import errno
import os
import sys
from contextlib import contextmanager
from typing import IO, Iterator, TextIO
def flush_stdout(stdout: TextIO, data: str) -> None:
    has_binary_io = hasattr(stdout, 'encoding') and hasattr(stdout, 'buffer')
    try:
        with _blocking_io(stdout):
            if has_binary_io:
                stdout.buffer.write(data.encode(stdout.encoding or 'utf-8', 'replace'))
            else:
                stdout.write(data)
            stdout.flush()
    except OSError as e:
        if e.args and e.args[0] == errno.EINTR:
            pass
        elif e.args and e.args[0] == 0:
            pass
        else:
            raise
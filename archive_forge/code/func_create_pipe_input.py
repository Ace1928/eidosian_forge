from __future__ import annotations
import io
import sys
from typing import ContextManager, TextIO
from .base import DummyInput, Input, PipeInput
def create_pipe_input() -> ContextManager[PipeInput]:
    """
    Create an input pipe.
    This is mostly useful for unit testing.

    Usage::

        with create_pipe_input() as input:
            input.send_text('inputdata')

    Breaking change: In prompt_toolkit 3.0.28 and earlier, this was returning
    the `PipeInput` directly, rather than through a context manager.
    """
    if sys.platform == 'win32':
        from .win32_pipe import Win32PipeInput
        return Win32PipeInput.create()
    else:
        from .posix_pipe import PosixPipeInput
        return PosixPipeInput.create()
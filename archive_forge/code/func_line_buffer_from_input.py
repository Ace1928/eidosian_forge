import contextlib
import enum
import io
import os
import signal
import subprocess
import sys
import types
import typing
from typing import Any, Optional, Type, Dict, TextIO
from autopage import command
def line_buffer_from_input(input_stream: Optional[typing.IO]=None) -> bool:
    """
    Return whether line buffering should be enabled for a given input stream.

    When data is being read from an input stream, processed somehow, and then
    written to an autopaged output stream, it may be desirable to enable line
    buffering on the output. This means that each line of data written to the
    output will be visible immediately, as opposed to waiting for the output
    buffer to fill up. This is, however, slower.

    If the input stream is a file, line buffering is unnecessary. This function
    determines whether an input stream might require line buffering on output.

    If no input stream is specified, sys.stdin is assumed.

        >>> with AutoPager(line_buffering=line_buffer_from_input()) as out:
        >>>     for l in sys.stdin:
        >>>         out.write(l)
    """
    if input_stream is None:
        input_stream = sys.stdin
    return not (input_stream.seekable() and (not input_stream.isatty()))
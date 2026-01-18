import os
import subprocess
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import IO, Generator, List, Optional, Tuple, Union
from .logging import get_logger
@contextmanager
def capture_output() -> Generator[StringIO, None, None]:
    """Capture output that is printed to terminal.

    Taken from https://stackoverflow.com/a/34738440

    Example:
    ```py
    >>> with capture_output() as output:
    ...     print("hello world")
    >>> assert output.getvalue() == "hello world
"
    ```
    """
    output = StringIO()
    previous_output = sys.stdout
    sys.stdout = output
    yield output
    sys.stdout = previous_output
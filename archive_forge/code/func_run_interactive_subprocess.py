import os
import subprocess
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import IO, Generator, List, Optional, Tuple, Union
from .logging import get_logger
@contextmanager
def run_interactive_subprocess(command: Union[str, List[str]], folder: Optional[Union[str, Path]]=None, **kwargs) -> Generator[Tuple[IO[str], IO[str]], None, None]:
    """Run a subprocess in an interactive mode in a context manager.

    Args:
        command (`str` or `List[str]`):
            The command to execute as a string or list of strings.
        folder (`str`, *optional*):
            The folder in which to run the command. Defaults to current working
            directory (from `os.getcwd()`).
        kwargs (`Dict[str]`):
            Keyword arguments to be passed to the `subprocess.run` underlying command.

    Returns:
        `Tuple[IO[str], IO[str]]`: A tuple with `stdin` and `stdout` to interact
        with the process (input and output are utf-8 encoded).

    Example:
    ```python
    with _interactive_subprocess("git credential-store get") as (stdin, stdout):
        # Write to stdin
        stdin.write("url=hf.co
username=obama
".encode("utf-8"))
        stdin.flush()

        # Read from stdout
        output = stdout.read().decode("utf-8")
    ```
    """
    if isinstance(command, str):
        command = command.split()
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace', cwd=folder or os.getcwd(), **kwargs) as process:
        assert process.stdin is not None, 'subprocess is opened as subprocess.PIPE'
        assert process.stdout is not None, 'subprocess is opened as subprocess.PIPE'
        yield (process.stdin, process.stdout)
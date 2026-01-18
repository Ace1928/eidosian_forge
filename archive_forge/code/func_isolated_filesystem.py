import contextlib
import io
import os
import shlex
import shutil
import sys
import tempfile
import typing as t
from types import TracebackType
from . import formatting
from . import termui
from . import utils
from ._compat import _find_binary_reader
@contextlib.contextmanager
def isolated_filesystem(self, temp_dir: t.Optional[t.Union[str, 'os.PathLike[str]']]=None) -> t.Iterator[str]:
    """A context manager that creates a temporary directory and
        changes the current working directory to it. This isolates tests
        that affect the contents of the CWD to prevent them from
        interfering with each other.

        :param temp_dir: Create the temporary directory under this
            directory. If given, the created directory is not removed
            when exiting.

        .. versionchanged:: 8.0
            Added the ``temp_dir`` parameter.
        """
    cwd = os.getcwd()
    dt = tempfile.mkdtemp(dir=temp_dir)
    os.chdir(dt)
    try:
        yield dt
    finally:
        os.chdir(cwd)
        if temp_dir is None:
            try:
                shutil.rmtree(dt)
            except OSError:
                pass
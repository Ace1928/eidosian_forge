import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def compatible_tags(python_version: Optional[PythonVersion]=None, interpreter: Optional[str]=None, platforms: Optional[Iterable[str]]=None) -> Iterator[Tag]:
    """
    Yields the sequence of tags that are compatible with a specific version of Python.

    The tags consist of:
    - py*-none-<platform>
    - <interpreter>-none-any  # ... if `interpreter` is provided.
    - py*-none-any
    """
    if not python_version:
        python_version = sys.version_info[:2]
    platforms = list(platforms or platform_tags())
    for version in _py_interpreter_range(python_version):
        for platform_ in platforms:
            yield Tag(version, 'none', platform_)
    if interpreter:
        yield Tag(interpreter, 'none', 'any')
    for version in _py_interpreter_range(python_version):
        yield Tag(version, 'none', 'any')
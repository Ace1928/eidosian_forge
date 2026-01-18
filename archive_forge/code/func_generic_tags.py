import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def generic_tags(interpreter: Optional[str]=None, abis: Optional[Iterable[str]]=None, platforms: Optional[Iterable[str]]=None, *, warn: bool=False) -> Iterator[Tag]:
    """
    Yields the tags for a generic interpreter.

    The tags consist of:
    - <interpreter>-<abi>-<platform>

    The "none" ABI will be added if it was not explicitly provided.
    """
    if not interpreter:
        interp_name = interpreter_name()
        interp_version = interpreter_version(warn=warn)
        interpreter = ''.join([interp_name, interp_version])
    if abis is None:
        abis = _generic_abi()
    else:
        abis = list(abis)
    platforms = list(platforms or platform_tags())
    if 'none' not in abis:
        abis.append('none')
    for abi in abis:
        for platform_ in platforms:
            yield Tag(interpreter, abi, platform_)
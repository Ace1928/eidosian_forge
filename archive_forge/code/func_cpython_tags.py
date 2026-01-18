import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def cpython_tags(python_version: Optional[PythonVersion]=None, abis: Optional[Iterable[str]]=None, platforms: Optional[Iterable[str]]=None, *, warn: bool=False) -> Iterator[Tag]:
    """
    Yields the tags for a CPython interpreter.

    The tags consist of:
    - cp<python_version>-<abi>-<platform>
    - cp<python_version>-abi3-<platform>
    - cp<python_version>-none-<platform>
    - cp<less than python_version>-abi3-<platform>  # Older Python versions down to 3.2.

    If python_version only specifies a major version then user-provided ABIs and
    the 'none' ABItag will be used.

    If 'abi3' or 'none' are specified in 'abis' then they will be yielded at
    their normal position and not at the beginning.
    """
    if not python_version:
        python_version = sys.version_info[:2]
    interpreter = f'cp{_version_nodot(python_version[:2])}'
    if abis is None:
        if len(python_version) > 1:
            abis = _cpython_abis(python_version, warn)
        else:
            abis = []
    abis = list(abis)
    for explicit_abi in ('abi3', 'none'):
        try:
            abis.remove(explicit_abi)
        except ValueError:
            pass
    platforms = list(platforms or platform_tags())
    for abi in abis:
        for platform_ in platforms:
            yield Tag(interpreter, abi, platform_)
    if _abi3_applies(python_version):
        yield from (Tag(interpreter, 'abi3', platform_) for platform_ in platforms)
    yield from (Tag(interpreter, 'none', platform_) for platform_ in platforms)
    if _abi3_applies(python_version):
        for minor_version in range(python_version[1] - 1, 1, -1):
            for platform_ in platforms:
                interpreter = 'cp{version}'.format(version=_version_nodot((python_version[0], minor_version)))
                yield Tag(interpreter, 'abi3', platform_)
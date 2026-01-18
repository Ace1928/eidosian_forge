import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def mac_platforms(version: Optional[MacVersion]=None, arch: Optional[str]=None) -> Iterator[str]:
    """
    Yields the platform tags for a macOS system.

    The `version` parameter is a two-item tuple specifying the macOS version to
    generate platform tags for. The `arch` parameter is the CPU architecture to
    generate platform tags for. Both parameters default to the appropriate value
    for the current system.
    """
    version_str, _, cpu_arch = platform.mac_ver()
    if version is None:
        version = cast('MacVersion', tuple(map(int, version_str.split('.')[:2])))
        if version == (10, 16):
            version_str = subprocess.run([sys.executable, '-sS', '-c', 'import platform; print(platform.mac_ver()[0])'], check=True, env={'SYSTEM_VERSION_COMPAT': '0'}, stdout=subprocess.PIPE, universal_newlines=True).stdout
            version = cast('MacVersion', tuple(map(int, version_str.split('.')[:2])))
    else:
        version = version
    if arch is None:
        arch = _mac_arch(cpu_arch)
    else:
        arch = arch
    if (10, 0) <= version and version < (11, 0):
        for minor_version in range(version[1], -1, -1):
            compat_version = (10, minor_version)
            binary_formats = _mac_binary_formats(compat_version, arch)
            for binary_format in binary_formats:
                yield 'macosx_{major}_{minor}_{binary_format}'.format(major=10, minor=minor_version, binary_format=binary_format)
    if version >= (11, 0):
        for major_version in range(version[0], 10, -1):
            compat_version = (major_version, 0)
            binary_formats = _mac_binary_formats(compat_version, arch)
            for binary_format in binary_formats:
                yield 'macosx_{major}_{minor}_{binary_format}'.format(major=major_version, minor=0, binary_format=binary_format)
    if version >= (11, 0):
        if arch == 'x86_64':
            for minor_version in range(16, 3, -1):
                compat_version = (10, minor_version)
                binary_formats = _mac_binary_formats(compat_version, arch)
                for binary_format in binary_formats:
                    yield 'macosx_{major}_{minor}_{binary_format}'.format(major=compat_version[0], minor=compat_version[1], binary_format=binary_format)
        else:
            for minor_version in range(16, 3, -1):
                compat_version = (10, minor_version)
                binary_format = 'universal2'
                yield 'macosx_{major}_{minor}_{binary_format}'.format(major=compat_version[0], minor=compat_version[1], binary_format=binary_format)
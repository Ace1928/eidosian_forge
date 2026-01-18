import asyncio
import logging
from concurrent.futures import Executor, ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
from multiprocessing import freeze_support
from typing import Set, Tuple
import click
import black
from _black_version import version as __version__
from black.concurrency import maybe_install_uvloop
def parse_python_variant_header(value: str) -> Tuple[bool, Set[black.TargetVersion]]:
    if value == 'pyi':
        return (True, set())
    else:
        versions = set()
        for version in value.split(','):
            if version.startswith('py'):
                version = version[len('py'):]
            if '.' in version:
                major_str, *rest = version.split('.')
            else:
                major_str = version[0]
                rest = [version[1:]] if len(version) > 1 else []
            try:
                major = int(major_str)
                if major not in (2, 3):
                    raise InvalidVariantHeader('major version must be 2 or 3')
                if len(rest) > 0:
                    minor = int(rest[0])
                    if major == 2:
                        raise InvalidVariantHeader('Python 2 is not supported')
                else:
                    minor = 7 if major == 2 else 3
                version_str = f'PY{major}{minor}'
                if major == 3 and (not hasattr(black.TargetVersion, version_str)):
                    raise InvalidVariantHeader(f'3.{minor} is not supported')
                versions.add(black.TargetVersion[version_str])
            except (KeyError, ValueError):
                raise InvalidVariantHeader("expected e.g. '3.7', 'py3.5'") from None
        return (False, versions)
import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def parse_tag(tag: str) -> FrozenSet[Tag]:
    """
    Parses the provided tag (e.g. `py3-none-any`) into a frozenset of Tag instances.

    Returning a set is required due to the possibility that the tag is a
    compressed tag set.
    """
    tags = set()
    interpreters, abis, platforms = tag.split('-')
    for interpreter in interpreters.split('.'):
        for abi in abis.split('.'):
            for platform_ in platforms.split('.'):
                tags.add(Tag(interpreter, abi, platform_))
    return frozenset(tags)
import contextlib
import functools
import os
import sys
from typing import TYPE_CHECKING, List, Optional, Type, cast
from pip._internal.utils.misc import strtobool
from .base import BaseDistribution, BaseEnvironment, FilesystemWheel, MemoryWheel, Wheel
def get_directory_distribution(directory: str) -> BaseDistribution:
    """Get the distribution metadata representation in the specified directory.

    This returns a Distribution instance from the chosen backend based on
    the given on-disk ``.dist-info`` directory.
    """
    return select_backend().Distribution.from_directory(directory)
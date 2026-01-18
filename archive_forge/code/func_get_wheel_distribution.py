import contextlib
import functools
import os
import sys
from typing import TYPE_CHECKING, List, Optional, Type, cast
from pip._internal.utils.misc import strtobool
from .base import BaseDistribution, BaseEnvironment, FilesystemWheel, MemoryWheel, Wheel
def get_wheel_distribution(wheel: Wheel, canonical_name: str) -> BaseDistribution:
    """Get the representation of the specified wheel's distribution metadata.

    This returns a Distribution instance from the chosen backend based on
    the given wheel's ``.dist-info`` directory.

    :param canonical_name: Normalized project name of the given wheel.
    """
    return select_backend().Distribution.from_wheel(wheel, canonical_name)
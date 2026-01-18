import errno
import os
from io import BytesIO
from typing import Set
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import bedding
def add_runtime_ignores(ignores):
    """Add some ignore patterns that only exists in memory.

    This is used by some plugins that want bzr to ignore files,
    but don't want to change a users ignore list.
    (Such as a conversion script that needs to ignore temporary files,
    but does not want to modify the project's ignore list.)

    :param ignores: A list or generator of ignore patterns.
    :return: None
    """
    global _runtime_ignores
    _runtime_ignores.update(set(ignores))
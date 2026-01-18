from gitdb.db.base import (
from gitdb.exc import (
from gitdb.stream import (
from gitdb.base import (
from gitdb.util import (
from gitdb.fun import (
from gitdb.utils.encoding import force_bytes
import tempfile
import os
import sys
def object_path(self, hexsha):
    """
        :return: path at which the object with the given hexsha would be stored,
            relative to the database root"""
    return join(hexsha[:2], hexsha[2:])
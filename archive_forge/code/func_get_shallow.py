import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
def get_shallow(self) -> Set[ObjectID]:
    """Get the set of shallow commits.

        Returns: Set of shallow commits.
        """
    f = self.get_named_file('shallow')
    if f is None:
        return set()
    with f:
        return {line.strip() for line in f}
import os
import stat
import sys
import warnings
from contextlib import suppress
from io import BytesIO
from typing import (
from .errors import NotTreeError
from .file import GitFile
from .objects import (
from .pack import (
from .protocol import DEPTH_INFINITE
from .refs import PEELED_TAG_SUFFIX, Ref
def add_pack_data(self, count: int, unpacked_objects: Iterator[UnpackedObject], progress=None) -> None:
    """Add pack data to this object store.

        Args:
          count: Number of items to add
          pack_data: Iterator over pack data tuples
        """
    for unpacked_object in unpacked_objects:
        self.add_object(unpacked_object.sha_file())
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
def get_blob_normalizer(self):
    """Return a BlobNormalizer object."""
    git_attributes = {}
    config_stack = self.get_config_stack()
    try:
        tree = self.object_store[self.refs[b'HEAD']].tree
        return TreeBlobNormalizer(config_stack, git_attributes, self.object_store, tree)
    except KeyError:
        return BlobNormalizer(config_stack, git_attributes)
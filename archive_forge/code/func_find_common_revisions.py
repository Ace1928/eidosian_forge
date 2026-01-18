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
def find_common_revisions(self, graphwalker):
    """Find which revisions this store has in common using graphwalker.

        Args:
          graphwalker: A graphwalker object.
        Returns: List of SHAs that are in common
        """
    haves = []
    sha = next(graphwalker)
    while sha:
        if sha in self:
            haves.append(sha)
            graphwalker.ack(sha)
        sha = next(graphwalker)
    return haves
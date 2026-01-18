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
def peel_sha(self, sha):
    """Peel all tags from a SHA.

        Args:
          sha: The object SHA to peel.
        Returns: The fully-peeled SHA1 of a tag object, after peeling all
            intermediate tags; if the original ref does not point to a tag,
            this will equal the original SHA1.
        """
    warnings.warn('Please use dulwich.object_store.peel_sha()', DeprecationWarning, stacklevel=2)
    return peel_sha(self, sha)[1]
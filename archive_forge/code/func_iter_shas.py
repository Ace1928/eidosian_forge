from io import BytesIO
import os
import stat
import sys
from dulwich.diff_tree import (
from dulwich.errors import (
from dulwich.file import GitFile
from dulwich.objects import (
from dulwich.pack import (
from dulwich.refs import ANNOTATED_TAG_SUFFIX
def iter_shas(self, shas):
    """Iterate over the objects for the specified shas.

        Args:
          shas: Iterable object with SHAs
        Returns: Object iterator
        """
    return ObjectStoreIterator(self, shas)
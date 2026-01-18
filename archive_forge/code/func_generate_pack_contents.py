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
def generate_pack_contents(self, have, want, shallow=None, progress=None):
    """Iterate over the contents of a pack file.

        Args:
          have: List of SHA1s of objects that should not be sent
          want: List of SHA1s of objects that should be sent
          shallow: Set of shallow commit SHA1s to skip
          progress: Optional progress reporting method
        """
    missing = self.find_missing_objects(have, want, shallow, progress)
    return self.iter_shas(missing)
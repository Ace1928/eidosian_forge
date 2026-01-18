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
def itershas(self):
    """Iterate over the SHAs."""
    for sha in self._shas:
        yield sha
    for sha in self.sha_iter:
        self._shas.append(sha)
        yield sha
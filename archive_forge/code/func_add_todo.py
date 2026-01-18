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
def add_todo(self, entries: Iterable[Tuple[ObjectID, Optional[bytes], Optional[int], bool]]):
    self.objects_to_send.update([e for e in entries if e[0] not in self.sha_done])
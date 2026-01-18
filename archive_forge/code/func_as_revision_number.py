from __future__ import annotations
from contextlib import contextmanager
import datetime
import os
import re
import shutil
import sys
from types import ModuleType
from typing import Any
from typing import cast
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import revision
from . import write_hooks
from .. import util
from ..runtime import migration
from ..util import compat
from ..util import not_none
def as_revision_number(self, id_: Optional[str]) -> Optional[Union[str, Tuple[str, ...]]]:
    """Convert a symbolic revision, i.e. 'head' or 'base', into
        an actual revision number."""
    with self._catch_revision_errors():
        rev, branch_name = self.revision_map._resolve_revision_number(id_)
    if not rev:
        return None
    elif id_ == 'heads':
        return rev
    else:
        return rev[0]
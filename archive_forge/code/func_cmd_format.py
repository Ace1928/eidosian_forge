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
def cmd_format(self, verbose: bool, include_branches: bool=False, include_doc: bool=False, include_parents: bool=False, tree_indicators: bool=True) -> str:
    if verbose:
        return self.log_entry
    else:
        return self._head_only(include_branches, include_doc, include_parents, tree_indicators)
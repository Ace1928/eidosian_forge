from __future__ import annotations
import contextlib
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import inspect
from . import compare
from . import render
from .. import util
from ..operations import ops
from ..util import sqla_compat
def run_no_autogenerate(self, rev: _GetRevArg, migration_context: MigrationContext) -> None:
    self._run_environment(rev, migration_context, False)
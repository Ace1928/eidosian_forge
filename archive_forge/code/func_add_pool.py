from __future__ import annotations
import collections
import re
import typing
from typing import Any
from typing import Dict
from typing import Optional
import warnings
import weakref
from . import config
from .util import decorator
from .util import gc_collect
from .. import event
from .. import pool
from ..util import await_only
from ..util.typing import Literal
def add_pool(self, pool):
    event.listen(pool, 'checkout', self._add_conn)
    event.listen(pool, 'checkin', self._remove_conn)
    event.listen(pool, 'close', self._remove_conn)
    event.listen(pool, 'close_detached', self._remove_conn)
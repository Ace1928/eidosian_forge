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
def all_dialects(exclude=None):
    import sqlalchemy.dialects as d
    for name in d.__all__:
        if exclude and name in exclude:
            continue
        mod = getattr(d, name, None)
        if not mod:
            mod = getattr(__import__('sqlalchemy.dialects.%s' % name).dialects, name)
        yield mod.dialect()
from __future__ import annotations
from argparse import Namespace
import collections
import inspect
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from . import mock
from . import requirements as _requirements
from .util import fail
from .. import util
@classmethod
def push_engine(cls, db, namespace):
    assert _current, "Can't push without a default Config set up"
    cls.push(Config(db, _current.db_opts, _current.options, _current.file_config), namespace)
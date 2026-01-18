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
def generate_cases(cls, argname, cases):
    case_names = [argname if c is True else 'not_' + argname if c is False else c for c in cases]
    typ = type(argname, (Variation,), {'__slots__': tuple(case_names)})
    return [typ(casename, argname, case_names) for casename in case_names]
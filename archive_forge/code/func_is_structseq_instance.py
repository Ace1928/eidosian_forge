from __future__ import annotations
import types
from collections.abc import Hashable
from typing import (
from typing_extensions import NamedTuple  # Generic NamedTuple: Python 3.11+
from typing_extensions import OrderedDict  # Generic OrderedDict: Python 3.7.2+
from typing_extensions import Self  # Python 3.11+
from typing_extensions import TypeAlias  # Python 3.10+
from typing_extensions import Final, Protocol, runtime_checkable  # Python 3.8+
from optree import _C
from optree._C import PyTreeKind, PyTreeSpec
from optree._C import (
def is_structseq_instance(obj: object) -> bool:
    """Return whether the object is an instance of PyStructSequence."""
    return is_structseq_class(type(obj))
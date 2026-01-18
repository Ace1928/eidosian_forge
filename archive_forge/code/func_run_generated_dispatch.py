from __future__ import annotations
from collections import deque
from enum import Enum
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util import langhelpers
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
def run_generated_dispatch(self, target: object, internal_dispatch: _TraverseInternalsType, generate_dispatcher_name: str) -> Any:
    dispatcher: _InternalTraversalDispatchType
    try:
        dispatcher = target.__class__.__dict__[generate_dispatcher_name]
    except KeyError:
        dispatcher = self.generate_dispatch(target.__class__, internal_dispatch, generate_dispatcher_name)
    return dispatcher(target, self)
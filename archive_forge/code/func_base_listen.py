from __future__ import annotations
import collections
import types
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
import weakref
from .. import exc
from .. import util
def base_listen(self, propagate: bool=False, insert: bool=False, named: bool=False, retval: Optional[bool]=None, asyncio: bool=False) -> None:
    target, identifier = (self.dispatch_target, self.identifier)
    dispatch_collection = getattr(target.dispatch, identifier)
    for_modify = dispatch_collection.for_modify(target.dispatch)
    if asyncio:
        for_modify._set_asyncio()
    if insert:
        for_modify.insert(self, propagate)
    else:
        for_modify.append(self, propagate)
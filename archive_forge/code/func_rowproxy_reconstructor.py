from __future__ import annotations
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type
def rowproxy_reconstructor(cls: Type[BaseRow], state: Dict[str, Any]) -> BaseRow:
    obj = cls.__new__(cls)
    obj.__setstate__(state)
    return obj
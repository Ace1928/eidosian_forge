from __future__ import annotations
import collections.abc as collections_abc
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import quote_plus
from urllib.parse import unquote
from .interfaces import Dialect
from .. import exc
from .. import util
from ..dialects import plugins
from ..dialects import registry
def get_dialect(self, _is_async: bool=False) -> Type[Dialect]:
    """Return the SQLAlchemy :class:`_engine.Dialect` class corresponding
        to this URL's driver name.

        """
    entrypoint = self._get_entrypoint()
    if _is_async:
        dialect_cls = entrypoint.get_async_dialect_cls(self)
    else:
        dialect_cls = entrypoint.get_dialect_cls(self)
    return dialect_cls
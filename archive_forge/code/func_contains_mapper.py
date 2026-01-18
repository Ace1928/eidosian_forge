from __future__ import annotations
from functools import reduce
from itertools import chain
import logging
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import base as orm_base
from ._typing import insp_is_mapper_property
from .. import exc
from .. import util
from ..sql import visitors
from ..sql.cache_key import HasCacheKey
def contains_mapper(self, mapper: Mapper[Any]) -> bool:
    _m_path = cast(_OddPathRepresentation, self.path)
    for path_mapper in [_m_path[i] for i in range(0, len(_m_path), 2)]:
        if path_mapper.mapper.isa(mapper):
            return True
    else:
        return False
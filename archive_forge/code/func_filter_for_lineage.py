from __future__ import annotations
import collections
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Deque
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Protocol
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy import util as sqlautil
from .. import util
from ..util import not_none
def filter_for_lineage(self, targets: Iterable[_TR], check_against: Optional[str], include_dependencies: bool=False) -> Tuple[_TR, ...]:
    id_, branch_label = self._resolve_revision_number(check_against)
    shares = []
    if branch_label:
        shares.append(branch_label)
    if id_:
        shares.extend(id_)
    return tuple((tg for tg in targets if self._shares_lineage(tg, shares, include_dependencies=include_dependencies)))
from __future__ import annotations
from typing import Any
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union
from ... import exc
from ... import util
from ...sql._typing import _DMLTableArgument
from ...sql.base import _exclusive_against
from ...sql.base import _generative
from ...sql.base import ColumnCollection
from ...sql.base import ReadOnlyColumnCollection
from ...sql.dml import Insert as StandardInsert
from ...sql.elements import ClauseElement
from ...sql.elements import KeyedColumnElement
from ...sql.expression import alias
from ...sql.selectable import NamedFromClause
from ...util.typing import Self
@util.memoized_property
def inserted_alias(self) -> NamedFromClause:
    return alias(self.table, name='inserted')
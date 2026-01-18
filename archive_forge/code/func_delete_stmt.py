from __future__ import annotations
from itertools import chain
from itertools import groupby
from itertools import zip_longest
import operator
from . import attributes
from . import exc as orm_exc
from . import loading
from . import sync
from .base import state_str
from .. import exc as sa_exc
from .. import future
from .. import sql
from .. import util
from ..engine import cursor as _cursor
from ..sql import operators
from ..sql.elements import BooleanClauseList
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
def delete_stmt():
    clauses = BooleanClauseList._construct_raw(operators.and_)
    for col in mapper._pks_by_table[table]:
        clauses._append_inplace(col == sql.bindparam(col.key, type_=col.type))
    if need_version_id:
        clauses._append_inplace(mapper.version_id_col == sql.bindparam(mapper.version_id_col.key, type_=mapper.version_id_col.type))
    return table.delete().where(clauses)
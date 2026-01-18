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
def post_update(base_mapper, states, uowtransaction, post_update_cols):
    """Issue UPDATE statements on behalf of a relationship() which
    specifies post_update.

    """
    states_to_update = list(_organize_states_for_post_update(base_mapper, states, uowtransaction))
    for table, mapper in base_mapper._sorted_tables.items():
        if table not in mapper._pks_by_table:
            continue
        update = ((state, state_dict, sub_mapper, connection, mapper._get_committed_state_attr_by_column(state, state_dict, mapper.version_id_col) if mapper.version_id_col is not None else None) for state, state_dict, sub_mapper, connection in states_to_update if table in sub_mapper._pks_by_table)
        update = _collect_post_update_commands(base_mapper, uowtransaction, table, update, post_update_cols)
        _emit_post_update_statements(base_mapper, uowtransaction, mapper, table, update)
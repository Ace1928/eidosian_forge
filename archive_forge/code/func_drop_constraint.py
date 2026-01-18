from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import Index
from sqlalchemy import MetaData
from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy import schema as sql_schema
from sqlalchemy import Table
from sqlalchemy import types as sqltypes
from sqlalchemy.sql.schema import SchemaEventTarget
from sqlalchemy.util import OrderedDict
from sqlalchemy.util import topological
from ..util import exc
from ..util.sqla_compat import _columns_for_constraint
from ..util.sqla_compat import _copy
from ..util.sqla_compat import _copy_expression
from ..util.sqla_compat import _ensure_scope_for_ddl
from ..util.sqla_compat import _fk_is_self_referential
from ..util.sqla_compat import _idx_table_bound_expressions
from ..util.sqla_compat import _insert_inline
from ..util.sqla_compat import _is_type_bound
from ..util.sqla_compat import _remove_column_from_collection
from ..util.sqla_compat import _resolve_for_variant
from ..util.sqla_compat import _select
from ..util.sqla_compat import constraint_name_defined
from ..util.sqla_compat import constraint_name_string
def drop_constraint(self, const: Constraint) -> None:
    if not const.name:
        raise ValueError('Constraint must have a name')
    try:
        if const.name in self.col_named_constraints:
            col, const = self.col_named_constraints.pop(const.name)
            for col_const in list(self.columns[col.name].constraints):
                if col_const.name == const.name:
                    self.columns[col.name].constraints.remove(col_const)
        elif constraint_name_string(const.name):
            const = self.named_constraints.pop(const.name)
        elif const in self.unnamed_constraints:
            self.unnamed_constraints.remove(const)
    except KeyError:
        if _is_type_bound(const):
            return
        raise ValueError("No such constraint: '%s'" % const.name)
    else:
        if isinstance(const, PrimaryKeyConstraint):
            for col in const.columns:
                self.columns[col.name].primary_key = False
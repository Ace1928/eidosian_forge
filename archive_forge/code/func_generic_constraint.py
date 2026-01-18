from __future__ import annotations
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import schema as sa_schema
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.schema import Constraint
from sqlalchemy.sql.schema import Index
from sqlalchemy.types import Integer
from sqlalchemy.types import NULLTYPE
from .. import util
from ..util import sqla_compat
def generic_constraint(self, name: Optional[sqla_compat._ConstraintNameDefined], table_name: str, type_: Optional[str], schema: Optional[str]=None, **kw) -> Any:
    t = self.table(table_name, schema=schema)
    types: Dict[Optional[str], Any] = {'foreignkey': lambda name: sa_schema.ForeignKeyConstraint([], [], name=name), 'primary': sa_schema.PrimaryKeyConstraint, 'unique': sa_schema.UniqueConstraint, 'check': lambda name: sa_schema.CheckConstraint('', name=name), None: sa_schema.Constraint}
    try:
        const = types[type_]
    except KeyError as ke:
        raise TypeError("'type' can be one of %s" % ', '.join(sorted((repr(x) for x in types)))) from ke
    else:
        const = const(name=name)
        t.append_constraint(const)
        return const
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
def primary_key_constraint(self, name: Optional[sqla_compat._ConstraintNameDefined], table_name: str, cols: Sequence[str], schema: Optional[str]=None, **dialect_kw) -> PrimaryKeyConstraint:
    m = self.metadata()
    columns = [sa_schema.Column(n, NULLTYPE) for n in cols]
    t = sa_schema.Table(table_name, m, *columns, schema=schema)
    p = sa_schema.PrimaryKeyConstraint(*[t.c[n] for n in cols], name=name, **dialect_kw)
    return p
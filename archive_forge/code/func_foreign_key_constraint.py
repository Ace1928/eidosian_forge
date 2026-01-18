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
def foreign_key_constraint(self, name: Optional[sqla_compat._ConstraintNameDefined], source: str, referent: str, local_cols: List[str], remote_cols: List[str], onupdate: Optional[str]=None, ondelete: Optional[str]=None, deferrable: Optional[bool]=None, source_schema: Optional[str]=None, referent_schema: Optional[str]=None, initially: Optional[str]=None, match: Optional[str]=None, **dialect_kw) -> ForeignKeyConstraint:
    m = self.metadata()
    if source == referent and source_schema == referent_schema:
        t1_cols = local_cols + remote_cols
    else:
        t1_cols = local_cols
        sa_schema.Table(referent, m, *[sa_schema.Column(n, NULLTYPE) for n in remote_cols], schema=referent_schema)
    t1 = sa_schema.Table(source, m, *[sa_schema.Column(n, NULLTYPE) for n in util.unique_list(t1_cols)], schema=source_schema)
    tname = '%s.%s' % (referent_schema, referent) if referent_schema else referent
    dialect_kw['match'] = match
    f = sa_schema.ForeignKeyConstraint(local_cols, ['%s.%s' % (tname, n) for n in remote_cols], name=name, onupdate=onupdate, ondelete=ondelete, deferrable=deferrable, initially=initially, **dialect_kw)
    t1.append_constraint(f)
    return f
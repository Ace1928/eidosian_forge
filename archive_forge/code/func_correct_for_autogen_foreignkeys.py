from __future__ import annotations
import re
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import schema
from sqlalchemy import types as sqltypes
from .base import alter_table
from .base import AlterColumn
from .base import ColumnDefault
from .base import ColumnName
from .base import ColumnNullable
from .base import ColumnType
from .base import format_column_name
from .base import format_server_default
from .impl import DefaultImpl
from .. import util
from ..util import sqla_compat
from ..util.sqla_compat import _is_mariadb
from ..util.sqla_compat import _is_type_bound
from ..util.sqla_compat import compiles
def correct_for_autogen_foreignkeys(self, conn_fks, metadata_fks):
    conn_fk_by_sig = {self._create_reflected_constraint_sig(fk).unnamed_no_options: fk for fk in conn_fks}
    metadata_fk_by_sig = {self._create_metadata_constraint_sig(fk).unnamed_no_options: fk for fk in metadata_fks}
    for sig in set(conn_fk_by_sig).intersection(metadata_fk_by_sig):
        mdfk = metadata_fk_by_sig[sig]
        cnfk = conn_fk_by_sig[sig]
        if mdfk.ondelete is not None and mdfk.ondelete.lower() == 'restrict' and (cnfk.ondelete is None):
            cnfk.ondelete = 'RESTRICT'
        if mdfk.onupdate is not None and mdfk.onupdate.lower() == 'restrict' and (cnfk.onupdate is None):
            cnfk.onupdate = 'RESTRICT'
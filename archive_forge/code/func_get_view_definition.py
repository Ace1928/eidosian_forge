from the proposed insertion. These values are specified using the
from __future__ import annotations
import datetime
import numbers
import re
from typing import Optional
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import text
from ... import types as sqltypes
from ... import util
from ...engine import default
from ...engine import processors
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import ColumnElement
from ...sql import compiler
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...types import BLOB  # noqa
from ...types import BOOLEAN  # noqa
from ...types import CHAR  # noqa
from ...types import DECIMAL  # noqa
from ...types import FLOAT  # noqa
from ...types import INTEGER  # noqa
from ...types import NUMERIC  # noqa
from ...types import REAL  # noqa
from ...types import SMALLINT  # noqa
from ...types import TEXT  # noqa
from ...types import TIMESTAMP  # noqa
from ...types import VARCHAR  # noqa
@reflection.cache
def get_view_definition(self, connection, view_name, schema=None, **kw):
    if schema is not None:
        qschema = self.identifier_preparer.quote_identifier(schema)
        master = f'{qschema}.sqlite_master'
        s = "SELECT sql FROM %s WHERE name = ? AND type='view'" % (master,)
        rs = connection.exec_driver_sql(s, (view_name,))
    else:
        try:
            s = "SELECT sql FROM  (SELECT * FROM sqlite_master UNION ALL   SELECT * FROM sqlite_temp_master) WHERE name = ? AND type='view'"
            rs = connection.exec_driver_sql(s, (view_name,))
        except exc.DBAPIError:
            s = "SELECT sql FROM sqlite_master WHERE name = ? AND type='view'"
            rs = connection.exec_driver_sql(s, (view_name,))
    result = rs.fetchall()
    if result:
        return result[0].sql
    else:
        raise exc.NoSuchTableError(f'{schema}.{view_name}' if schema else view_name)
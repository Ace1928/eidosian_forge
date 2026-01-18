from __future__ import annotations
from collections import defaultdict
import contextlib
from copy import copy
from itertools import filterfalse
import re
import sys
import warnings
from . import assertsql
from . import config
from . import engines
from . import mock
from .exclusions import db_spec
from .util import fail
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import types as sqltypes
from .. import util
from ..engine import default
from ..engine import url
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import decorator
def assert_tables_equal(self, table, reflected_table, strict_types=False, strict_constraints=True):
    assert len(table.c) == len(reflected_table.c)
    for c, reflected_c in zip(table.c, reflected_table.c):
        eq_(c.name, reflected_c.name)
        assert reflected_c is reflected_table.c[c.name]
        if strict_constraints:
            eq_(c.primary_key, reflected_c.primary_key)
            eq_(c.nullable, reflected_c.nullable)
        if strict_types:
            msg = "Type '%s' doesn't correspond to type '%s'"
            assert isinstance(reflected_c.type, type(c.type)), msg % (reflected_c.type, c.type)
        else:
            self.assert_types_base(reflected_c, c)
        if isinstance(c.type, sqltypes.String):
            eq_(c.type.length, reflected_c.type.length)
        if strict_constraints:
            eq_({f.column.name for f in c.foreign_keys}, {f.column.name for f in reflected_c.foreign_keys})
        if c.server_default:
            assert isinstance(reflected_c.server_default, schema.FetchedValue)
    if strict_constraints:
        assert len(table.primary_key) == len(reflected_table.primary_key)
        for c in table.primary_key:
            assert reflected_table.primary_key.columns[c.name] is not None
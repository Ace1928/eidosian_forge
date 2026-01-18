from __future__ import annotations
import configparser
from contextlib import contextmanager
import io
import re
from typing import Any
from typing import Dict
from sqlalchemy import Column
from sqlalchemy import inspect
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import testing
from sqlalchemy import text
from sqlalchemy.testing import config
from sqlalchemy.testing import mock
from sqlalchemy.testing.assertions import eq_
from sqlalchemy.testing.fixtures import TablesTest as SQLAlchemyTablesTest
from sqlalchemy.testing.fixtures import TestBase as SQLAlchemyTestBase
import alembic
from .assertions import _get_dialect
from ..environment import EnvironmentContext
from ..migration import MigrationContext
from ..operations import Operations
from ..util import sqla_compat
from ..util.sqla_compat import create_mock_engine
from ..util.sqla_compat import sqla_14
from ..util.sqla_compat import sqla_2
def op_fixture(dialect='default', as_sql=False, naming_convention=None, literal_binds=False, native_boolean=None):
    opts = {}
    if naming_convention:
        opts['target_metadata'] = MetaData(naming_convention=naming_convention)

    class buffer_:

        def __init__(self):
            self.lines = []

        def write(self, msg):
            msg = msg.strip()
            msg = re.sub('[\\n\\t]', '', msg)
            if as_sql:
                msg = re.sub('    ', '', msg)
                msg = re.sub('\\;\\n*$', '', msg)
            self.lines.append(msg)

        def flush(self):
            pass
    buf = buffer_()

    class ctx(MigrationContext):

        def get_buf(self):
            return buf

        def clear_assertions(self):
            buf.lines[:] = []

        def assert_(self, *sql):
            eq_(buf.lines, [re.sub('[\\n\\t]', '', s) for s in sql])

        def assert_contains(self, sql):
            for stmt in buf.lines:
                if re.sub('[\\n\\t]', '', sql) in stmt:
                    return
            else:
                assert False, 'Could not locate fragment %r in %r' % (sql, buf.lines)
    if as_sql:
        opts['as_sql'] = as_sql
    if literal_binds:
        opts['literal_binds'] = literal_binds
    if not sqla_14 and dialect == 'mariadb':
        ctx_dialect = _get_dialect('mysql')
        ctx_dialect.server_version_info = (10, 4, 0, 'MariaDB')
    else:
        ctx_dialect = _get_dialect(dialect)
    if native_boolean is not None:
        ctx_dialect.supports_native_boolean = native_boolean
        ctx_dialect.non_native_boolean_check_constraint = True
    if not as_sql:

        def execute(stmt, *multiparam, **param):
            if isinstance(stmt, str):
                stmt = text(stmt)
            assert stmt.supports_execution
            sql = str(stmt.compile(dialect=ctx_dialect))
            buf.write(sql)
        connection = mock.Mock(dialect=ctx_dialect, execute=execute)
    else:
        opts['output_buffer'] = buf
        connection = None
    context = ctx(ctx_dialect, connection, opts)
    alembic.op._proxy = Operations(context)
    return context
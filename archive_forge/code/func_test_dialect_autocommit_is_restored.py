import importlib
from . import testing
from .. import assert_raises
from .. import config
from .. import engines
from .. import eq_
from .. import fixtures
from .. import is_not_none
from .. import is_true
from .. import ne_
from .. import provide_metadata
from ..assertions import expect_raises
from ..assertions import expect_raises_message
from ..config import requirements
from ..provision import set_default_schema_on_connection
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import dialects
from ... import event
from ... import exc
from ... import Integer
from ... import literal_column
from ... import select
from ... import String
from ...sql.compiler import Compiled
from ...util import inspect_getfullargspec
@testing.requires.independent_readonly_connections
@testing.variation('use_dialect_setting', [True, False])
def test_dialect_autocommit_is_restored(self, testing_engine, use_dialect_setting):
    """test #10147"""
    if use_dialect_setting:
        e = testing_engine(options={'isolation_level': 'AUTOCOMMIT'})
    else:
        e = testing_engine().execution_options(isolation_level='AUTOCOMMIT')
    levels = requirements.get_isolation_levels(config)
    default = levels['default']
    with e.connect() as conn:
        self._test_conn_autocommits(conn, True)
    with e.connect() as conn:
        conn.execution_options(isolation_level=default)
        self._test_conn_autocommits(conn, False)
    with e.connect() as conn:
        self._test_conn_autocommits(conn, True)
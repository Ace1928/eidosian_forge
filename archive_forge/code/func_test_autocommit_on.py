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
def test_autocommit_on(self, connection_no_trans):
    conn = connection_no_trans
    c2 = conn.execution_options(isolation_level='AUTOCOMMIT')
    self._test_conn_autocommits(c2, True)
    c2.dialect.reset_isolation_level(c2.connection.dbapi_connection)
    self._test_conn_autocommits(conn, False)
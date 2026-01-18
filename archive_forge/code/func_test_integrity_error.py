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
@requirements.duplicate_key_raises_integrity_error
def test_integrity_error(self):
    with config.db.connect() as conn:
        trans = conn.begin()
        conn.execute(self.tables.manual_pk.insert(), {'id': 1, 'data': 'd1'})
        assert_raises(exc.IntegrityError, conn.execute, self.tables.manual_pk.insert(), {'id': 1, 'data': 'd1'})
        trans.rollback()
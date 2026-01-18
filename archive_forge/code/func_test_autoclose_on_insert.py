from decimal import Decimal
import uuid
from . import testing
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import Double
from ... import Float
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import Numeric
from ... import select
from ... import String
from ...types import LargeBinary
from ...types import UUID
from ...types import Uuid
@requirements.autoincrement_insert
def test_autoclose_on_insert(self, connection):
    r = connection.execute(self.tables.autoinc_pk.insert(), dict(data='some data'))
    assert r._soft_closed
    assert not r.closed
    assert r.is_insert
    assert not r.returns_rows or r.fetchone() is None
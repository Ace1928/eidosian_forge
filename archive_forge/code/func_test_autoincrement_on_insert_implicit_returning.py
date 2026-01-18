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
def test_autoincrement_on_insert_implicit_returning(self, connection):
    connection.execute(self.tables.autoinc_pk.insert(), dict(data='some data'))
    self._assert_round_trip(self.tables.autoinc_pk, connection)
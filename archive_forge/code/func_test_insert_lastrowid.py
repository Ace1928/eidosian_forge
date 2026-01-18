from .. import config
from .. import fixtures
from ..assertions import eq_
from ..assertions import is_true
from ..config import requirements
from ..provision import normalize_sequence
from ..schema import Column
from ..schema import Table
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import Sequence
from ... import String
from ... import testing
def test_insert_lastrowid(self, connection):
    r = connection.execute(self.tables.seq_pk.insert(), dict(data='some data'))
    eq_(r.inserted_primary_key, (testing.db.dialect.default_sequence_base,))
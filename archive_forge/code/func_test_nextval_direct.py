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
def test_nextval_direct(self, connection):
    r = connection.scalar(self.tables.seq_pk.c.id.default)
    eq_(r, testing.db.dialect.default_sequence_base)
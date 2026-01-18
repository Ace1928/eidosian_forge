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
@testing.requires.schemas
def test_has_sequence_remote_not_in_default(self, connection):
    eq_(inspect(connection).has_sequence('schema_seq'), False)
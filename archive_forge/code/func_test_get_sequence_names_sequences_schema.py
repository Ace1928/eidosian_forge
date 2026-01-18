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
def test_get_sequence_names_sequences_schema(self, connection):
    eq_(sorted(inspect(connection).get_sequence_names(schema=config.test_schema)), ['schema_seq', 'user_id_seq'])
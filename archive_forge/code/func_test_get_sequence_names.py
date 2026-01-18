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
def test_get_sequence_names(self, connection):
    exp = {'other_seq', 'user_id_seq'}
    res = set(inspect(connection).get_sequence_names())
    is_true(res.intersection(exp) == exp)
    is_true('schema_seq' not in res)
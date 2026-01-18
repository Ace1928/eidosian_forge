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
def test_nextval_direct_schema_translate(self, connection):
    seq = normalize_sequence(config, Sequence('noret_sch_id_seq', schema='alt_schema'))
    connection = connection.execution_options(schema_translate_map={'alt_schema': config.test_schema})
    r = connection.scalar(seq)
    eq_(r, testing.db.dialect.default_sequence_base)
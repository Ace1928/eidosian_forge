import random
from . import testing
from .. import config
from .. import fixtures
from .. import util
from ..assertions import eq_
from ..assertions import is_false
from ..assertions import is_true
from ..config import requirements
from ..schema import Table
from ... import CheckConstraint
from ... import Column
from ... import ForeignKeyConstraint
from ... import Index
from ... import inspect
from ... import Integer
from ... import schema
from ... import String
from ... import UniqueConstraint
@requirements.create_table
@requirements.schemas
@util.provide_metadata
def test_create_table_schema(self):
    table = self._simple_fixture(schema=config.test_schema)
    table.create(config.db, checkfirst=False)
    self._simple_roundtrip(table)
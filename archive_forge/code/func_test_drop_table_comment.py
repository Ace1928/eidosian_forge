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
@requirements.comment_reflection
@util.provide_metadata
def test_drop_table_comment(self, connection):
    table = self._simple_fixture()
    table.create(connection, checkfirst=False)
    table.comment = 'a comment'
    connection.execute(schema.SetTableComment(table))
    connection.execute(schema.DropTableComment(table))
    eq_(inspect(connection).get_table_comment('test_table'), {'text': None})
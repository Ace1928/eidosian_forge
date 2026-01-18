import importlib
from . import testing
from .. import assert_raises
from .. import config
from .. import engines
from .. import eq_
from .. import fixtures
from .. import is_not_none
from .. import is_true
from .. import ne_
from .. import provide_metadata
from ..assertions import expect_raises
from ..assertions import expect_raises_message
from ..config import requirements
from ..provision import set_default_schema_on_connection
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import dialects
from ... import event
from ... import exc
from ... import Integer
from ... import literal_column
from ... import select
from ... import String
from ...sql.compiler import Compiled
from ...util import inspect_getfullargspec
@testing.fixture
def multirow_fixture(self, metadata, connection):
    mytable = Table('mytable', metadata, Column('myid', Integer), Column('name', String(50)), Column('desc', String(50)))
    mytable.create(connection)
    connection.execute(mytable.insert(), [{'myid': 1, 'name': 'a', 'desc': 'a_desc'}, {'myid': 2, 'name': 'b', 'desc': 'b_desc'}, {'myid': 3, 'name': 'c', 'desc': 'c_desc'}, {'myid': 4, 'name': 'd', 'desc': 'd_desc'}])
    yield mytable
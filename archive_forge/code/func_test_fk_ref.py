import operator
import re
import sqlalchemy as sa
from .. import config
from .. import engines
from .. import eq_
from .. import expect_raises
from .. import expect_raises_message
from .. import expect_warnings
from .. import fixtures
from .. import is_
from ..provision import get_temp_table_name
from ..provision import temp_table_keyword_args
from ..schema import Column
from ..schema import Table
from ... import event
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import String
from ... import testing
from ... import types as sql_types
from ...engine import Inspector
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...exc import NoSuchTableError
from ...exc import UnreflectableTableError
from ...schema import DDL
from ...schema import Index
from ...sql.elements import quoted_name
from ...sql.schema import BLANK_SCHEMA
from ...testing import ComparesIndexes
from ...testing import ComparesTables
from ...testing import is_false
from ...testing import is_true
from ...testing import mock
@testing.combinations(('id',), ('(3)',), ('col%p',), ('[brack]',), argnames='columnname')
@testing.variation('use_composite', [True, False])
@testing.combinations(('plain',), ('(2)',), ('per % cent',), ('[brackets]',), argnames='tablename')
def test_fk_ref(self, connection, metadata, use_composite, tablename, columnname):
    tt = Table(tablename, metadata, Column(columnname, Integer, key='id', primary_key=True), test_needs_fk=True)
    if use_composite:
        tt.append_column(Column('id2', Integer, primary_key=True))
    if use_composite:
        Table('other', metadata, Column('id', Integer, primary_key=True), Column('ref', Integer), Column('ref2', Integer), sa.ForeignKeyConstraint(['ref', 'ref2'], [tt.c.id, tt.c.id2]), test_needs_fk=True)
    else:
        Table('other', metadata, Column('id', Integer, primary_key=True), Column('ref', ForeignKey(tt.c.id)), test_needs_fk=True)
    metadata.create_all(connection)
    m2 = MetaData()
    o2 = Table('other', m2, autoload_with=connection)
    t1 = m2.tables[tablename]
    assert o2.c.ref.references(t1.c[0])
    if use_composite:
        assert o2.c.ref2.references(t1.c[1])
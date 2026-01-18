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
@testing.combinations((None, 'CASCADE', None, testing.requires.foreign_key_constraint_option_reflection_ondelete), (None, None, 'SET NULL', testing.requires.foreign_key_constraint_option_reflection_onupdate), ({}, None, 'NO ACTION', testing.requires.foreign_key_constraint_option_reflection_onupdate), ({}, 'NO ACTION', None, testing.requires.fk_constraint_option_reflection_ondelete_noaction), (None, None, 'RESTRICT', testing.requires.fk_constraint_option_reflection_onupdate_restrict), (None, 'RESTRICT', None, testing.requires.fk_constraint_option_reflection_ondelete_restrict), argnames='expected,ondelete,onupdate')
def test_get_foreign_key_options(self, connection, metadata, expected, ondelete, onupdate):
    options = {}
    if ondelete:
        options['ondelete'] = ondelete
    if onupdate:
        options['onupdate'] = onupdate
    if expected is None:
        expected = options
    Table('x', metadata, Column('id', Integer, primary_key=True), test_needs_fk=True)
    Table('table', metadata, Column('id', Integer, primary_key=True), Column('x_id', Integer, ForeignKey('x.id', name='xid')), Column('test', String(10)), test_needs_fk=True)
    Table('user', metadata, Column('id', Integer, primary_key=True), Column('name', String(50), nullable=False), Column('tid', Integer), sa.ForeignKeyConstraint(['tid'], ['table.id'], name='myfk', **options), test_needs_fk=True)
    metadata.create_all(connection)
    insp = inspect(connection)
    opts = insp.get_foreign_keys('table')[0]['options']
    eq_({k: opts[k] for k in opts if opts[k]}, {})
    opts = insp.get_foreign_keys('user')[0]['options']
    eq_(opts, expected)
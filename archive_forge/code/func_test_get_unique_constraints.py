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
@testing.combinations((True, testing.requires.schemas), (False,), argnames='use_schema')
@testing.requires.unique_constraint_reflection
def test_get_unique_constraints(self, metadata, connection, use_schema):
    if use_schema:
        schema = config.test_schema
    else:
        schema = None
    uniques = sorted([{'name': 'unique_a', 'column_names': ['a']}, {'name': 'unique_a_b_c', 'column_names': ['a', 'b', 'c']}, {'name': 'unique_c_a_b', 'column_names': ['c', 'a', 'b']}, {'name': 'unique_asc_key', 'column_names': ['asc', 'key']}, {'name': 'i.have.dots', 'column_names': ['b']}, {'name': 'i have spaces', 'column_names': ['c']}], key=operator.itemgetter('name'))
    table = Table('testtbl', metadata, Column('a', sa.String(20)), Column('b', sa.String(30)), Column('c', sa.Integer), Column('asc', sa.String(30)), Column('key', sa.String(30)), schema=schema)
    for uc in uniques:
        table.append_constraint(sa.UniqueConstraint(*uc['column_names'], name=uc['name']))
    table.create(connection)
    insp = inspect(connection)
    reflected = sorted(insp.get_unique_constraints('testtbl', schema=schema), key=operator.itemgetter('name'))
    names_that_duplicate_index = set()
    eq_(len(uniques), len(reflected))
    for orig, refl in zip(uniques, reflected):
        dupe = refl.pop('duplicates_index', None)
        if dupe:
            names_that_duplicate_index.add(dupe)
        eq_(refl.pop('comment', None), None)
        eq_(orig, refl)
    reflected_metadata = MetaData()
    reflected = Table('testtbl', reflected_metadata, autoload_with=connection, schema=schema)
    idx_names = {idx.name for idx in reflected.indexes}
    uq_names = {uq.name for uq in reflected.constraints if isinstance(uq, sa.UniqueConstraint)}.difference(['unique_c_a_b'])
    assert not idx_names.intersection(uq_names)
    if names_that_duplicate_index:
        eq_(names_that_duplicate_index, idx_names)
        eq_(uq_names, set())
    no_cst = self.tables.no_constraints.name
    eq_(insp.get_unique_constraints(no_cst, schema=schema), [])
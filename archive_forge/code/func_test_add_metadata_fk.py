from sqlalchemy import Column
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from ._autogen_fixtures import AutogenFixtureTest
from ...testing import combinations
from ...testing import config
from ...testing import eq_
from ...testing import mock
from ...testing import TestBase
def test_add_metadata_fk(self):
    m1 = MetaData()
    m2 = MetaData()
    Table('ref', m1, Column('id', Integer, primary_key=True))
    Table('t', m1, Column('x', Integer), Column('y', Integer))
    ref = Table('ref', m2, Column('id', Integer, primary_key=True))
    t2 = Table('t', m2, Column('x', Integer), Column('y', Integer))
    t2.append_constraint(ForeignKeyConstraint([t2.c.x], [ref.c.id], name='fk1'))
    t2.append_constraint(ForeignKeyConstraint([t2.c.y], [ref.c.id], name='fk2'))

    def include_object(object_, name, type_, reflected, compare_to):
        return not (isinstance(object_, ForeignKeyConstraint) and type_ == 'foreign_key_constraint' and (not reflected) and (name == 'fk1'))
    diffs = self._fixture(m1, m2, object_filters=include_object)
    self._assert_fk_diff(diffs[0], 'add_fk', 't', ['y'], 'ref', ['id'], name='fk2')
    eq_(len(diffs), 1)
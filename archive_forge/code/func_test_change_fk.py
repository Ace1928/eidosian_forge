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
@combinations(('object',), ('name',))
@config.requirements.no_name_normalize
def test_change_fk(self, hook_type):
    m1 = MetaData()
    m2 = MetaData()
    r1a = Table('ref_a', m1, Column('a', Integer, primary_key=True))
    Table('ref_b', m1, Column('a', Integer, primary_key=True), Column('b', Integer, primary_key=True))
    t1 = Table('t', m1, Column('x', Integer), Column('y', Integer), Column('z', Integer))
    t1.append_constraint(ForeignKeyConstraint([t1.c.x], [r1a.c.a], name='fk1'))
    t1.append_constraint(ForeignKeyConstraint([t1.c.y], [r1a.c.a], name='fk2'))
    Table('ref_a', m2, Column('a', Integer, primary_key=True))
    r2b = Table('ref_b', m2, Column('a', Integer, primary_key=True), Column('b', Integer, primary_key=True))
    t2 = Table('t', m2, Column('x', Integer), Column('y', Integer), Column('z', Integer))
    t2.append_constraint(ForeignKeyConstraint([t2.c.x, t2.c.z], [r2b.c.a, r2b.c.b], name='fk1'))
    t2.append_constraint(ForeignKeyConstraint([t2.c.y, t2.c.z], [r2b.c.a, r2b.c.b], name='fk2'))
    if hook_type == 'object':

        def include_object(object_, name, type_, reflected, compare_to):
            return not (isinstance(object_, ForeignKeyConstraint) and type_ == 'foreign_key_constraint' and (name == 'fk1'))
        diffs = self._fixture(m1, m2, object_filters=include_object)
    elif hook_type == 'name':

        def include_name(name, type_, parent_names):
            if type_ == 'index':
                return True
            if name == 'fk1':
                eq_(type_, 'foreign_key_constraint')
                eq_(parent_names, {'schema_name': None, 'table_name': 't', 'schema_qualified_table_name': 't'})
                return False
            else:
                return True
        diffs = self._fixture(m1, m2, name_filters=include_name)
    if hook_type == 'object':
        self._assert_fk_diff(diffs[0], 'remove_fk', 't', ['y'], 'ref_a', ['a'], name='fk2')
        self._assert_fk_diff(diffs[1], 'add_fk', 't', ['y', 'z'], 'ref_b', ['a', 'b'], name='fk2')
        eq_(len(diffs), 2)
    elif hook_type == 'name':
        eq_({(d[0], d[1].name) for d in diffs}, {('add_fk', 'fk2'), ('add_fk', 'fk1'), ('remove_fk', 'fk2')})
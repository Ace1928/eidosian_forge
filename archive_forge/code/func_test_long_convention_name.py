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
@testing.combinations(('fk',), ('pk',), ('ix',), ('ck', testing.requires.check_constraint_reflection.as_skips()), ('uq', testing.requires.unique_constraint_reflection.as_skips()), argnames='type_')
def test_long_convention_name(self, type_, metadata, connection):
    actual_name, reflected_name = getattr(self, type_)(metadata, connection)
    assert len(actual_name) > 255
    if reflected_name is not None:
        overlap = actual_name[0:len(reflected_name)]
        if len(overlap) < len(actual_name):
            eq_(overlap[0:-5], reflected_name[0:len(overlap) - 5])
        else:
            eq_(overlap, reflected_name)
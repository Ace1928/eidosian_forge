import datetime
from .. import engines
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import DateTime
from ... import func
from ... import Integer
from ... import select
from ... import sql
from ... import String
from ... import testing
from ... import text
@classmethod
def define_tables(cls, metadata):
    cls.tables.percent_table = Table('percent%table', metadata, Column('percent%', Integer), Column('spaces % more spaces', Integer))
    cls.tables.lightweight_percent_table = sql.table('percent%table', sql.column('percent%'), sql.column('spaces % more spaces'))
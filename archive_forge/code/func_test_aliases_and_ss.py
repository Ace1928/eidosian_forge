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
def test_aliases_and_ss(self):
    engine = self._fixture(False)
    s1 = select(sql.literal_column('1').label('x')).execution_options(stream_results=True).subquery()
    with engine.begin() as conn:
        result = conn.execute(s1.select())
        assert not self._is_server_side(result.cursor)
        result.close()
    s2 = select(1).select_from(s1)
    with engine.begin() as conn:
        result = conn.execute(s2)
        assert not self._is_server_side(result.cursor)
        result.close()
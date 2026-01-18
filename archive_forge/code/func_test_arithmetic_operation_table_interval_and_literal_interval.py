import datetime
import decimal
import json
import re
import uuid
from .. import config
from .. import engines
from .. import fixtures
from .. import mock
from ..assertions import eq_
from ..assertions import is_
from ..assertions import ne_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import and_
from ... import ARRAY
from ... import BigInteger
from ... import bindparam
from ... import Boolean
from ... import case
from ... import cast
from ... import Date
from ... import DateTime
from ... import Float
from ... import Integer
from ... import Interval
from ... import JSON
from ... import literal
from ... import literal_column
from ... import MetaData
from ... import null
from ... import Numeric
from ... import select
from ... import String
from ... import testing
from ... import Text
from ... import Time
from ... import TIMESTAMP
from ... import type_coerce
from ... import TypeDecorator
from ... import Unicode
from ... import UnicodeText
from ... import UUID
from ... import Uuid
from ...orm import declarative_base
from ...orm import Session
from ...sql import sqltypes
from ...sql.sqltypes import LargeBinary
from ...sql.sqltypes import PickleType
def test_arithmetic_operation_table_interval_and_literal_interval(self, connection, arithmetic_table_fixture):
    interval_table = arithmetic_table_fixture
    data = datetime.timedelta(days=2, seconds=5)
    connection.execute(interval_table.insert(), {'id': 1, 'interval_data': data})
    value = connection.execute(select(interval_table.c.interval_data - literal(self.data))).scalar()
    eq_(value, data - self.data)
    value = connection.execute(select(interval_table.c.interval_data + literal(self.data))).scalar()
    eq_(value, data + self.data)
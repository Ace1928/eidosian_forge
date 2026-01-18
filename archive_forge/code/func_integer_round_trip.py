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
@testing.fixture
def integer_round_trip(self, metadata, connection):

    def run(datatype, data):
        int_table = Table('integer_table', metadata, Column('id', Integer, primary_key=True, test_needs_autoincrement=True), Column('integer_data', datatype))
        metadata.create_all(config.db)
        connection.execute(int_table.insert(), {'id': 1, 'integer_data': data})
        row = connection.execute(select(int_table.c.integer_data)).first()
        eq_(row, (data,))
        assert isinstance(row[0], int)
    return run
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
@testing.requires.precision_numerics_enotation_large
def test_enotation_decimal(self, do_numeric_test):
    """test exceedingly small decimals.

        Decimal reports values with E notation when the exponent
        is greater than 6.

        """
    numbers = {decimal.Decimal('1E-2'), decimal.Decimal('1E-3'), decimal.Decimal('1E-4'), decimal.Decimal('1E-5'), decimal.Decimal('1E-6'), decimal.Decimal('1E-7'), decimal.Decimal('1E-8'), decimal.Decimal('0.01000005940696'), decimal.Decimal('0.00000005940696'), decimal.Decimal('0.00000000000696'), decimal.Decimal('0.70000000000696'), decimal.Decimal('696E-12')}
    do_numeric_test(Numeric(precision=18, scale=14), numbers, numbers)
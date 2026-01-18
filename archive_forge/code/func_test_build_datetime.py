import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_build_datetime(self):
    testtuples = (((DateTuple('2019', '06', '05', None, None, None), TimeTuple('01', '03', '11.858714', None)), datetime.datetime(2019, 6, 5, hour=1, minute=3, second=11, microsecond=858714)), ((DateTuple('1234', '02', '03', None, None, None), TimeTuple('23', '21', '28.512400', None)), datetime.datetime(1234, 2, 3, hour=23, minute=21, second=28, microsecond=512400)), ((DateTuple('1981', '04', '05', None, None, None), TimeTuple('23', '21', '28.512400', TimezoneTuple(False, None, '11', '15', '+11:15'))), datetime.datetime(1981, 4, 5, hour=23, minute=21, second=28, microsecond=512400, tzinfo=UTCOffset(name='+11:15', minutes=675))))
    for testtuple in testtuples:
        result = PythonTimeBuilder.build_datetime(*testtuple[0])
        self.assertEqual(result, testtuple[1])
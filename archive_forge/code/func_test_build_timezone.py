import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_build_timezone(self):
    testtuples = (({'Z': True, 'name': 'Z'}, datetime.timedelta(hours=0), 'UTC'), ({'negative': False, 'hh': '00', 'mm': '00', 'name': '+00:00'}, datetime.timedelta(hours=0), '+00:00'), ({'negative': False, 'hh': '01', 'mm': '00', 'name': '+01:00'}, datetime.timedelta(hours=1), '+01:00'), ({'negative': True, 'hh': '01', 'mm': '00', 'name': '-01:00'}, -datetime.timedelta(hours=1), '-01:00'), ({'negative': False, 'hh': '00', 'mm': '12', 'name': '+00:12'}, datetime.timedelta(minutes=12), '+00:12'), ({'negative': False, 'hh': '01', 'mm': '23', 'name': '+01:23'}, datetime.timedelta(hours=1, minutes=23), '+01:23'), ({'negative': True, 'hh': '01', 'mm': '23', 'name': '-01:23'}, -datetime.timedelta(hours=1, minutes=23), '-01:23'), ({'negative': False, 'hh': '00', 'name': '+00'}, datetime.timedelta(hours=0), '+00'), ({'negative': False, 'hh': '01', 'name': '+01'}, datetime.timedelta(hours=1), '+01'), ({'negative': True, 'hh': '01', 'name': '-01'}, -datetime.timedelta(hours=1), '-01'), ({'negative': False, 'hh': '12', 'name': '+12'}, datetime.timedelta(hours=12), '+12'), ({'negative': True, 'hh': '12', 'name': '-12'}, -datetime.timedelta(hours=12), '-12'))
    for testtuple in testtuples:
        result = PythonTimeBuilder.build_timezone(**testtuple[0])
        self.assertEqual(result.utcoffset(None), testtuple[1])
        self.assertEqual(result.tzname(None), testtuple[2])
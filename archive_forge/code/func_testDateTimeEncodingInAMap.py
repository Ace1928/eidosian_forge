import base64
import datetime
import json
import sys
import unittest
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testDateTimeEncodingInAMap(self):
    msg = MapToDateTimeValue(additionalProperties=[MapToDateTimeValue.AdditionalProperty(key='1st', value=datetime.datetime(2014, 7, 2, 23, 33, 25, 541000, tzinfo=util.TimeZoneOffset(datetime.timedelta(0)))), MapToDateTimeValue.AdditionalProperty(key='2nd', value=datetime.datetime(2015, 7, 2, 23, 33, 25, 541000, tzinfo=util.TimeZoneOffset(datetime.timedelta(0))))])
    self.assertEqual('{"1st": "2014-07-02T23:33:25.541000+00:00", "2nd": "2015-07-02T23:33:25.541000+00:00"}', encoding.MessageToJson(msg))
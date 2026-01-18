import datetime
import json
import math
import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import extra_types
def testDateField(self):

    class DateMsg(messages.Message):
        start_date = extra_types.DateField(1)
        all_dates = extra_types.DateField(2, repeated=True)
    msg = DateMsg(start_date=datetime.date(1752, 9, 9), all_dates=[datetime.date(1979, 5, 6), datetime.date(1980, 10, 24), datetime.date(1981, 1, 19)])
    msg_dict = {'start_date': '1752-09-09', 'all_dates': ['1979-05-06', '1980-10-24', '1981-01-19']}
    self.assertEqual(msg_dict, json.loads(encoding.MessageToJson(msg)))
    self.assertEqual(msg, encoding.JsonToMessage(DateMsg, json.dumps(msg_dict)))
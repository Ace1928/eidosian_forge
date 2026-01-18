import datetime
import logging
import time
from unittest import mock
import iso8601
from oslotest import base as test_base
from testtools import matchers
from oslo_utils import timeutils
def test_compare_micros(self):
    zulu = timeutils.parse_isotime('2012-02-14T20:53:07.6544')
    east = timeutils.parse_isotime('2012-02-14T19:53:07.654321-01:00')
    west = timeutils.parse_isotime('2012-02-14T21:53:07.655+01:00')
    self.assertTrue(east < west)
    self.assertTrue(east < zulu)
    self.assertTrue(zulu < west)
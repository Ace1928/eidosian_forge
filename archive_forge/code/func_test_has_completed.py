import os
import sys
import time
import random
import urllib
import datetime
import unittest
import threading
from unittest import mock
import requests
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.google import (
def test_has_completed(self):
    body1 = {'endTime': '2013-06-26T10:05:07.630-07:00', 'id': '3681664092089171723', 'kind': 'compute#operation', 'status': 'DONE', 'targetId': '16211908079305042870'}
    body2 = {'endTime': '2013-06-26T10:05:07.630-07:00', 'id': '3681664092089171723', 'kind': 'compute#operation', 'status': 'RUNNING', 'targetId': '16211908079305042870'}
    response1 = MockJsonResponse(body1)
    response2 = MockJsonResponse(body2)
    self.assertTrue(self.conn.has_completed(response1))
    self.assertFalse(self.conn.has_completed(response2))
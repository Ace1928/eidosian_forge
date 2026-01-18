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
def test_get_poll_request_kwargs(self):
    body = {'endTime': '2013-06-26T10:05:07.630-07:00', 'id': '3681664092089171723', 'kind': 'compute#operation', 'selfLink': 'https://www.googleapis.com/operations-test'}
    response = MockJsonResponse(body)
    expected_kwargs = {'action': 'https://www.googleapis.com/operations-test'}
    kwargs = self.conn.get_poll_request_kwargs(response, None, {})
    self.assertEqual(kwargs, expected_kwargs)
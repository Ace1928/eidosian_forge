import base64
import datetime
import sys
import contextlib
import unittest
import six
from six.moves import http_client
from six.moves import urllib_parse
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testOverwritesTransferUrlBase(self):
    client = self.__GetFakeClient()
    client.overwrite_transfer_urls_with_client_base = True
    client._url = 'http://custom.p.googleapis.com/'
    observed = client.FinalizeTransferUrl('http://normal.googleapis.com/path')
    expected = 'http://custom.p.googleapis.com/path'
    self.assertEqual(observed, expected)
import string
import unittest
import httplib2
import json
import mock
import six
from six.moves import http_client
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py import gzip
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
def testComputeEndByte(self):
    total_size = 100
    chunksize = 10
    download = transfer.Download.FromStream(six.StringIO(), chunksize=chunksize, total_size=total_size)
    self.assertEqual(chunksize - 1, download._Download__ComputeEndByte(0, end=50))
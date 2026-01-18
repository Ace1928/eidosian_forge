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
def testComputeEndByteNoTotal(self):
    download = transfer.Download.FromStream(six.StringIO())
    default_chunksize = download.chunksize
    for chunksize in (100, default_chunksize):
        download.chunksize = chunksize
        for start in (0, 10):
            self.assertEqual(download.chunksize + start - 1, download._Download__ComputeEndByte(start), msg='Failed on start={0}, chunksize={1}'.format(start, chunksize))
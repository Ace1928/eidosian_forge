import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def testSmallGzipObject(self):
    request = self.__GetRequest('zero-gzipd.html')
    self.__GetFile(request)
    self.assertEqual(0, self.__buffer.tell())
    additional_headers = {'accept-encoding': 'gzip, deflate'}
    self.__download.StreamInChunks(additional_headers=additional_headers)
    self.assertEqual(0, self.__buffer.tell())
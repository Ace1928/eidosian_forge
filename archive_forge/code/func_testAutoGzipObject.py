import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def testAutoGzipObject(self):
    request = storage.StorageObjectsGetRequest(bucket='ottenl-gzip', object='50K.txt')
    self.__GetFile(request)
    self.assertEqual(0, self.__buffer.tell())
    self.__download.StreamInChunks()
    self.assertEqual(50000, self.__buffer.tell())
    self.__ResetDownload(auto_transfer=True)
    self.__GetFile(request)
    self.assertEqual(50000, self.__buffer.tell())
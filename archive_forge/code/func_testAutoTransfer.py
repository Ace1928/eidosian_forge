import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def testAutoTransfer(self):
    self.__ResetDownload(auto_transfer=True)
    self.__GetFile(self.__GetRequest('fifteen_byte_file'))
    file_contents = self.__GetTestdataFileContents('fifteen_byte_file')
    self.assertEqual(15, self.__buffer.tell())
    self.__buffer.seek(0)
    self.assertEqual(file_contents, self.__buffer.read())
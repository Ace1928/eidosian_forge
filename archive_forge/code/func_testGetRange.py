import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def testGetRange(self):
    file_contents = self.__GetTestdataFileContents('fifteen_byte_file')
    self.__GetFile(self.__GetRequest('fifteen_byte_file'))
    self.__download.GetRange(5, 10)
    self.assertEqual(6, self.__buffer.tell())
    self.__buffer.seek(0)
    self.assertEqual(file_contents[5:11], self.__buffer.read())
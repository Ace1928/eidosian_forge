import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def testGetRangeWithNegativeStart(self):
    file_contents = self.__GetTestdataFileContents('fifteen_byte_file')
    self.__GetFile(self.__GetRequest('fifteen_byte_file'))
    self.__download.GetRange(-3)
    self.assertEqual(3, self.__buffer.tell())
    self.__buffer.seek(0)
    self.assertEqual(file_contents[-3:], self.__buffer.read())
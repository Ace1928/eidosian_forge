import json
import os
import random
import string
import unittest
import six
from apitools.base.py import transfer
import storage
def testSimpleUpload(self):
    filename = 'fifteen_byte_file'
    self.__ResetUpload(15)
    response = self.__InsertFile(filename)
    self.assertEqual(15, response.size)
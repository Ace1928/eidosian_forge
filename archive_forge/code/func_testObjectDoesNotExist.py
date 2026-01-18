import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def testObjectDoesNotExist(self):
    self.__ResetDownload(auto_transfer=True)
    with self.assertRaises(exceptions.HttpError):
        self.__GetFile(self.__GetRequest('nonexistent_file'))
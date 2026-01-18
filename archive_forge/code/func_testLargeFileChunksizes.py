import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def testLargeFileChunksizes(self):
    request = self.__GetRequest('thirty_meg_file')
    for chunksize in (1048576, 40 * 1048576):
        self.__ResetDownload()
        self.__download.chunksize = chunksize
        self.__GetAndStream(request)
        self.__buffer.seek(0)
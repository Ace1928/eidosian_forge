import os
import difflib
import unittest
import six
from apitools.gen import gen_client
from apitools.gen import test_utils
def testGenClient_DnsDoc(self):
    self._CheckGeneratedFiles('dns', 'v1')
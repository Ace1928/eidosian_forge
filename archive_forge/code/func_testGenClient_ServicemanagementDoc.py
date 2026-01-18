import os
import difflib
import unittest
import six
from apitools.gen import gen_client
from apitools.gen import test_utils
def testGenClient_ServicemanagementDoc(self):
    self._CheckGeneratedFiles('servicemanagement', 'v1')
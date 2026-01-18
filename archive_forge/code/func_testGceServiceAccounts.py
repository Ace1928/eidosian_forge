import json
import os.path
import shutil
import tempfile
import unittest
import mock
import six
from apitools.base.py import credentials_lib
from apitools.base.py import util
def testGceServiceAccounts(self):
    scopes = ['scope1']
    self._GetServiceCreds(service_account_name=None, scopes=None)
    self._GetServiceCreds(service_account_name=None, scopes=scopes)
    self._GetServiceCreds(service_account_name='my_service_account', scopes=scopes)
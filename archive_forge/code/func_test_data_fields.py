import os
import os.path
import stat
import unittest
from fixtures import MockPatch, TempDir
from testtools import TestCase
from lazr.restfulclient.authorize.oauth import (
def test_data_fields(self):
    access_token = AccessToken('key', 'secret', 'context')
    self.assertEqual(access_token.key, 'key')
    self.assertEqual(access_token.secret, 'secret')
    self.assertEqual(access_token.context, 'context')
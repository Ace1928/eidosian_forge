import os
import os.path
import stat
import unittest
from fixtures import MockPatch, TempDir
from testtools import TestCase
from lazr.restfulclient.authorize.oauth import (
def test_default_context(self):
    access_token = AccessToken('key', 'secret')
    self.assertIsNone(access_token.context)
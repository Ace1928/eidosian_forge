import os
import os.path
import stat
import unittest
from fixtures import MockPatch, TempDir
from testtools import TestCase
from lazr.restfulclient.authorize.oauth import (
def test_default_application_name(self):
    consumer = Consumer('key', 'secret')
    self.assertEqual(consumer.application_name, None)
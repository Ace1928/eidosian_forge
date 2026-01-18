from unittest import mock
import uuid
from designateclient import exceptions
from designateclient.tests import base
from designateclient import utils
def test_find_resourceid_with_multiple_resources(self):
    self.assertRaises(exceptions.NoUniqueMatch, self._find_resourceid_by_name_or_id, 'baba', by_name=True)
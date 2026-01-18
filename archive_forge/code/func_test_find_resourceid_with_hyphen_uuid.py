from unittest import mock
import uuid
from designateclient import exceptions
from designateclient.tests import base
from designateclient import utils
def test_find_resourceid_with_hyphen_uuid(self):
    expected = str(uuid.uuid4())
    observed = self._find_resourceid_by_name_or_id(expected)
    self.assertEqual(expected, observed)
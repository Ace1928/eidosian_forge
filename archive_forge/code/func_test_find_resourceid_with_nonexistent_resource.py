from unittest import mock
import uuid
from designateclient import exceptions
from designateclient.tests import base
from designateclient import utils
def test_find_resourceid_with_nonexistent_resource(self):
    self.assertRaises(exceptions.ResourceNotFound, self._find_resourceid_by_name_or_id, 'taz', by_name=True)
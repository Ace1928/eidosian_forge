from barbicanclient import client
from barbicanclient.tests import test_client
from barbicanclient.tests.utils import mock_delete_secret_for_responses
from barbicanclient.tests.utils import mock_get_secret_for_client
from barbicanclient.tests.v1.test_secrets import SecretData
from barbicanclient.v1 import secrets
from oslo_serialization import jsonutils
def test_list_consumers_from_secret_without_consumers(self):
    consumer_list = self._list_consumers(self.entity_href)
    self.assertTrue(len(consumer_list) == 0)
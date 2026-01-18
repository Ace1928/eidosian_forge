from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
def test_list_post(self):
    self.manager._list('/human_resources', 'human_resources', json={'id': 42})
    self.manager.client.post.assert_called_with('/human_resources', json={'id': 42})
import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def test_list_without_key_mapping(self):
    data_ = {'v1': '1', 'v2': '2'}
    body_ = {self.response_key: data_}
    url_ = 'test_url_post'
    self.manager.api.client.post = mock.Mock(return_value=(url_, body_))
    li = self.manager._list('url', self.response_key, obj_class, 'something')
    self.assertEqual(len(data_), len(li))
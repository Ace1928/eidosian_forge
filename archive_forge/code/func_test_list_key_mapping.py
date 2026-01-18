import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def test_list_key_mapping(self):
    data_ = {'values': ['p1', 'p2']}
    body_ = {self.response_key: data_}
    url_ = 'test_url_post'
    self.manager.api.client.post = mock.Mock(return_value=(url_, body_))
    li = self.manager._list('url', self.response_key, obj_class, 'something')
    data = data_['values']
    self.assertEqual(len(data), len(li))
    for i in range(0, len(li)):
        self.assertEqual(data[i], li[i])
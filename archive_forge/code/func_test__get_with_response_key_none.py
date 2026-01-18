import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def test__get_with_response_key_none(self):
    manager = self._get_mock()
    url_ = 'test-url'
    body_ = 'test-body'
    resp_ = 'test-resp'
    manager.api.client.get = mock.Mock(return_value=(resp_, body_))
    r = manager._get(url=url_, response_key=None)
    self.assertEqual(body_, r)
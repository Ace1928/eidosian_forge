import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def test_list_with_body_none(self):
    body = None
    li = self.manager._list('url', self.response_key, obj_class, body)
    self.assertEqual(len(self.data_g), len(li))
    for i in range(0, len(li)):
        self.assertEqual(self.data_g[i], li[i])
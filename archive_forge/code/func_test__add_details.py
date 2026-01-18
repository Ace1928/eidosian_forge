import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def test__add_details(self):
    robj = self.get_mock_resource_obj()
    info_ = {'name': 'test-human-id', 'test_attr': 5}
    robj._add_details(info_)
    self.assertEqual(info_['name'], robj.name)
    self.assertEqual(info_['test_attr'], robj.test_attr)
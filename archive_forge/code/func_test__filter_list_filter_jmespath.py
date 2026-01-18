from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test__filter_list_filter_jmespath(self):
    el1 = dict(id=100, name='donald', other='duck')
    el2 = dict(id=200, name='donald', other='trump')
    data = [el1, el2]
    ret = _utils._filter_list(data, 'donald', '[?other == `duck`]')
    self.assertEqual([el1], ret)
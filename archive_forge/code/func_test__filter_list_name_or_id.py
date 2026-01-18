from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test__filter_list_name_or_id(self):
    el1 = dict(id=100, name='donald')
    el2 = dict(id=200, name='pluto')
    data = [el1, el2]
    ret = _utils._filter_list(data, 'donald', None)
    self.assertEqual([el1], ret)
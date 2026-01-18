from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test__filter_list_dict2(self):
    el1 = dict(id=100, name='donald', last='duck', other=dict(category='duck', financial=dict(status='poor')))
    el2 = dict(id=200, name='donald', last='trump', other=dict(category='human', financial=dict(status='rich')))
    el3 = dict(id=300, name='donald', last='ronald mac', other=dict(category='clown', financial=dict(status='rich')))
    data = [el1, el2, el3]
    ret = _utils._filter_list(data, 'donald', {'other': {'financial': {'status': 'rich'}}})
    self.assertEqual([el2, el3], ret)
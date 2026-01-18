import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_cant_do_anything(self):

    class Test(resource.Resource):
        allow_create = False
        allow_fetch = False
        allow_commit = False
        allow_delete = False
        allow_head = False
        allow_list = False
    sot = Test()
    self.assertRaises(exceptions.MethodNotSupported, sot.create, '')
    self.assertRaises(exceptions.MethodNotSupported, sot.fetch, '')
    self.assertRaises(exceptions.MethodNotSupported, sot.delete, '')
    self.assertRaises(exceptions.MethodNotSupported, sot.head, '')
    the_list = sot.list('')
    self.assertRaises(exceptions.MethodNotSupported, next, the_list)
    sot._body = mock.Mock()
    sot._body.dirty = mock.Mock(return_value={'x': 'y'})
    self.assertRaises(exceptions.MethodNotSupported, sot.commit, '')
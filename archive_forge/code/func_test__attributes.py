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
def test__attributes(self):

    class Test(resource.Resource):
        foo = resource.Header('foo')
        bar = resource.Body('bar', aka='_bar')
        bar_local = resource.Body('bar_remote')
    sot = Test()
    self.assertEqual(sorted(['foo', 'bar', '_bar', 'bar_local', 'id', 'name', 'location']), sorted(sot._attributes()))
    self.assertEqual(sorted(['foo', 'bar', 'bar_local', 'id', 'name', 'location']), sorted(sot._attributes(include_aliases=False)))
    self.assertEqual(sorted(['foo', 'bar', '_bar', 'bar_remote', 'id', 'name', 'location']), sorted(sot._attributes(remote_names=True)))
    self.assertEqual(sorted(['bar', '_bar', 'bar_local', 'id', 'name', 'location']), sorted(sot._attributes(components=tuple([resource.Body, resource.Computed]))))
    self.assertEqual(('foo',), tuple(sot._attributes(components=tuple([resource.Header]))))
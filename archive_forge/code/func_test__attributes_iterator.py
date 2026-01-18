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
def test__attributes_iterator(self):

    class Parent(resource.Resource):
        foo = resource.Header('foo')
        bar = resource.Body('bar', aka='_bar')

    class Child(Parent):
        foo1 = resource.Header('foo1')
        bar1 = resource.Body('bar1')
    sot = Child()
    expected = ['foo', 'bar', 'foo1', 'bar1']
    for attr, component in sot._attributes_iterator():
        if attr in expected:
            expected.remove(attr)
    self.assertEqual([], expected)
    expected = ['foo', 'foo1']
    for attr, component in sot._attributes_iterator(components=tuple([resource.Header])):
        if attr in expected:
            expected.remove(attr)
    self.assertEqual([], expected)
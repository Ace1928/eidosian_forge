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
def test__consume_attrs(self):
    serverside_key1 = 'someKey1'
    clientside_key1 = 'some_key1'
    serverside_key2 = 'someKey2'
    clientside_key2 = 'some_key2'
    value1 = 'value1'
    value2 = 'value2'
    mapping = {serverside_key1: clientside_key1, serverside_key2: clientside_key2}
    other_key = 'otherKey'
    other_value = 'other'
    attrs = {clientside_key1: value1, serverside_key2: value2, other_key: other_value}
    sot = resource.Resource()
    result = sot._consume_attrs(mapping, attrs)
    self.assertDictEqual({other_key: other_value}, attrs)
    self.assertDictEqual({serverside_key1: value1, serverside_key2: value2}, result)
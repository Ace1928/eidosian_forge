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
def test_from_munch_existing(self):

    class Test(resource.Resource):
        attr = resource.Body('body_attr')
    value = 'value'
    orig = utils.Munch(body_attr=value)
    sot = Test._from_munch(orig)
    self.assertNotIn('body_attr', sot._body.dirty)
    self.assertEqual(value, sot.attr)
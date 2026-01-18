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
def test__alternate_id_from_other_property(self):

    class Test(resource.Resource):
        foo = resource.Body('foo')
        bar = resource.Body('bar', alternate_id=True)
    self.assertEqual('bar', Test._alternate_id())
    sot = Test(bar='bunnies')
    self.assertEqual(sot.id, 'bunnies')
    self.assertEqual(sot.bar, 'bunnies')
    sot = Test(id='chickens', bar='bunnies')
    self.assertEqual(sot.id, 'chickens')
    self.assertEqual(sot.bar, 'bunnies')
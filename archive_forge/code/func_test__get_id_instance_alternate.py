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
def test__get_id_instance_alternate(self):

    class Test(resource.Resource):
        attr = resource.Body('attr', alternate_id=True)
    value = 'id'
    sot = Test(attr=value)
    self.assertEqual(value, sot._get_id(sot))
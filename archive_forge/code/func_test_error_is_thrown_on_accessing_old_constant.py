import sys
import unittest
from libcloud.pricing import clear_pricing_data
from libcloud.utils.py3 import httplib, method_type
from libcloud.test.secrets import RACKSPACE_PARAMS, RACKSPACE_NOVA_PARAMS
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.compute.drivers.rackspace import RackspaceNodeDriver, RackspaceFirstGenNodeDriver
from libcloud.test.compute.test_openstack import (
def test_error_is_thrown_on_accessing_old_constant(self):
    for provider in DEPRECATED_RACKSPACE_PROVIDERS:
        try:
            get_driver(provider)
        except Exception as e:
            self.assertTrue(str(e).find('has been removed') != -1)
        else:
            self.fail('Exception was not thrown')
import contextlib
import datetime
from unittest import mock
import uuid
import warnings
from openstack.block_storage.v3 import volume
from openstack.compute.v2 import _proxy
from openstack.compute.v2 import aggregate
from openstack.compute.v2 import availability_zone as az
from openstack.compute.v2 import extension
from openstack.compute.v2 import flavor
from openstack.compute.v2 import hypervisor
from openstack.compute.v2 import image
from openstack.compute.v2 import keypair
from openstack.compute.v2 import migration
from openstack.compute.v2 import quota_set
from openstack.compute.v2 import server
from openstack.compute.v2 import server_action
from openstack.compute.v2 import server_group
from openstack.compute.v2 import server_interface
from openstack.compute.v2 import server_ip
from openstack.compute.v2 import server_migration
from openstack.compute.v2 import server_remote_console
from openstack.compute.v2 import service
from openstack.compute.v2 import usage
from openstack.compute.v2 import volume_attachment
from openstack import resource
from openstack.tests.unit import test_proxy_base
from openstack import warnings as os_warnings
def test_delete_flavor_extra_specs_prop(self):
    self._verify('openstack.compute.v2.flavor.Flavor.delete_extra_specs_property', self.proxy.delete_flavor_extra_specs_property, method_args=['value', 'prop'], expected_args=[self.proxy, 'prop'])
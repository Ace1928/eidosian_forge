from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_incompatible_microversion(self):
    self.session.default_microversion = '1.28'
    self.assertRaises(exceptions.NotSupported, self.node.inject_nmi, self.session)
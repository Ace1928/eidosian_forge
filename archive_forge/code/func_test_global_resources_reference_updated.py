import re
from unittest import mock
from oslo_config import cfg
from oslo_db import options
from oslotest import base
from neutron_lib.api import attributes
from neutron_lib.api.definitions import port
from neutron_lib.callbacks import registry
from neutron_lib.db import model_base
from neutron_lib.db import resource_extend
from neutron_lib import fixture
from neutron_lib.placement import client as place_client
from neutron_lib.plugins import directory
from neutron_lib.tests.unit.api import test_attributes
def test_global_resources_reference_updated(self):
    resources_ref = attributes.RESOURCES
    apis = fixture.APIDefinitionFixture()
    apis.setUp()
    attributes.RESOURCES['test_resource'] = {}
    self.assertIn('test_resource', resources_ref)
    attributes.RESOURCES[port.COLLECTION_NAME]['test_port_attr'] = {}
    self.assertIn('test_port_attr', attributes.RESOURCES[port.COLLECTION_NAME])
    apis.cleanUp()
    self.assertNotIn('test_port_attr', attributes.RESOURCES[port.COLLECTION_NAME])
    self.assertNotIn('test_resource', resources_ref)
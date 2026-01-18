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
def test_api_def_reference_updated(self):
    api_def_ref = port.RESOURCE_ATTRIBUTE_MAP
    apis = fixture.APIDefinitionFixture()
    apis.setUp()
    port.RESOURCE_ATTRIBUTE_MAP[port.COLLECTION_NAME]['test_attr'] = {}
    self.assertIn('test_attr', api_def_ref[port.COLLECTION_NAME])
    apis.cleanUp()
    self.assertNotIn('test_attr', port.RESOURCE_ATTRIBUTE_MAP[port.COLLECTION_NAME])
    self.assertNotIn('test_attr', api_def_ref[port.COLLECTION_NAME])
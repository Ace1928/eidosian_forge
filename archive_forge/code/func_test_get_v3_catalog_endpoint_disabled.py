import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_get_v3_catalog_endpoint_disabled(self):
    """Get back only enabled endpoints when get the v3 catalog."""
    enabled_endpoint_ref = self._create_endpoints()[1]
    user_id = uuid.uuid4().hex
    catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, self.project_bar['id'])
    endpoint_ids = [x['id'] for x in catalog[0]['endpoints']]
    self.assertEqual([enabled_endpoint_ref['id']], endpoint_ids)
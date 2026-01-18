import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_list_users_with_idp_id_protocol_id(self):
    federated_dict = unit.new_federated_user_ref()
    filters = ['idp_id', 'protocol_id']
    self._test_list_users_with_attribute(filters, federated_dict)
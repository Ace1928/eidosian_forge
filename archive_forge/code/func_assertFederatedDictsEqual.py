import uuid
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity.shadow_users import test_backend
from keystone.tests.unit.identity.shadow_users import test_core
from keystone.tests.unit.ksfixtures import database
def assertFederatedDictsEqual(self, fed_dict, fed_object):
    self.assertEqual(fed_dict['idp_id'], fed_object['idp_id'])
    self.assertEqual(fed_dict['protocol_id'], fed_object['protocols'][0]['protocol_id'])
    self.assertEqual(fed_dict['unique_id'], fed_object['protocols'][0]['unique_id'])
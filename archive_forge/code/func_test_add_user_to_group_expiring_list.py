import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_log import log
import sqlalchemy
from sqlalchemy import exc
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import sql
from keystone.common.sql import core
import keystone.conf
from keystone.credential.providers import fernet as credential_provider
from keystone import exception
from keystone.identity.backends import sql_model as identity_sql
from keystone.resource.backends import base as resource
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.limit import test_backends as limit_tests
from keystone.tests.unit.policy import test_backends as policy_tests
from keystone.tests.unit.resource import test_backends as resource_tests
from keystone.tests.unit.trust import test_backends as trust_tests
from keystone.trust.backends import sql as trust_sql
def test_add_user_to_group_expiring_list(self):
    self._build_fed_resource()
    domain = self._get_domain_fixture()
    self.config_fixture.config(group='federation', default_authorization_ttl=5)
    time = datetime.datetime.utcnow()
    tick = datetime.timedelta(minutes=5)
    new_group = unit.new_group_ref(domain_id=domain['id'])
    new_group = PROVIDERS.identity_api.create_group(new_group)
    exp_new_group = unit.new_group_ref(domain_id=domain['id'])
    exp_new_group = PROVIDERS.identity_api.create_group(exp_new_group)
    fed_dict = unit.new_federated_user_ref()
    fed_dict['idp_id'] = 'myidp'
    fed_dict['protocol_id'] = 'mapped'
    new_user = PROVIDERS.shadow_users_api.create_federated_user(domain['id'], fed_dict)
    PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
    PROVIDERS.identity_api.check_user_in_group(new_user['id'], new_group['id'])
    with freezegun.freeze_time(time - tick) as frozen_time:
        PROVIDERS.shadow_users_api.add_user_to_group_expires(new_user['id'], exp_new_group['id'])
        PROVIDERS.identity_api.check_user_in_group(new_user['id'], new_group['id'])
        groups = PROVIDERS.identity_api.list_groups_for_user(new_user['id'])
        self.assertEqual(len(groups), 2)
        for group in groups:
            if group.get('membership_expires_at'):
                self.assertEqual(group['membership_expires_at'], time)
        frozen_time.tick(tick)
        groups = PROVIDERS.identity_api.list_groups_for_user(new_user['id'])
        self.assertEqual(len(groups), 1)
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
def test_list_entities_filtered_by_domain(self):
    self.addCleanup(self.clean_up_entities)
    self.domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(self.domain1['id'], self.domain1)
    self.entity_list = {}
    self.domain1_entity_list = {}
    for entity in ['user', 'group', 'project']:
        DOMAIN1_ENTITIES = 3
        self.entity_list[entity] = self._create_test_data(entity, 2)
        self.domain1_entity_list[entity] = self._create_test_data(entity, DOMAIN1_ENTITIES, self.domain1['id'])
        hints = driver_hints.Hints()
        hints.add_filter('domain_id', self.domain1['id'])
        entities = self._list_entities(entity)(hints=hints)
        self.assertEqual(DOMAIN1_ENTITIES, len(entities))
        self._match_with_list(entities, self.domain1_entity_list[entity])
        self.assertFalse(hints.get_exact_filter_by_name('domain_id'))
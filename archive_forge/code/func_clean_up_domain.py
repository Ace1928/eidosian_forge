import copy
from unittest import mock
import uuid
from oslo_config import cfg
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def clean_up_domain(self):
    self.domain['enabled'] = False
    PROVIDERS.resource_api.update_domain(self.domain['id'], self.domain)
    PROVIDERS.resource_api.delete_domain(self.domain['id'])
    del self.domain
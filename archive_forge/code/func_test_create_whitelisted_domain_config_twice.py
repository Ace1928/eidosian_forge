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
def test_create_whitelisted_domain_config_twice(self):
    self._create_domain_config_twice(False)
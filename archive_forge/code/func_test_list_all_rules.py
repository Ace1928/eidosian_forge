import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_all_rules(self):
    self._setup_prior_two_implied()
    self._assert_two_rules_defined()
    self._delete_implied_role(self.prior, self.implied2)
    self._assert_one_rule_defined()
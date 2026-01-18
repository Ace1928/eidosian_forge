import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def test_MFA_multi_method_rules_requirements_met_succeeds(self):
    rule_list = [['password', 'totp']]
    totp_cred = self._create_totp_cred()
    self._update_user_with_MFA_rules(rule_list=rule_list)
    time = datetime.datetime.utcnow() + datetime.timedelta(seconds=5)
    with freezegun.freeze_time(time):
        auth_req = self.build_authentication_request(user_id=self.user_id, password=self.user['password'], user_domain_id=self.domain_id, passcode=totp._generate_totp_passcodes(totp_cred['blob'])[0])
        self.v3_create_token(auth_req)
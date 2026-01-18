import os
import subprocess
from unittest import mock
import uuid
from oslo_policy import policy as common_policy
from keystone.common import policies
from keystone.common.rbac_enforcer import policy
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_ignore_case_role_check(self):
    lowercase_action = 'example:lowercase_admin'
    uppercase_action = 'example:uppercase_admin'
    admin_credentials = {'roles': ['AdMiN']}
    policy.enforce(admin_credentials, lowercase_action, self.target)
    policy.enforce(admin_credentials, uppercase_action, self.target)
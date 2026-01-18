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
def test_early_AND_enforcement(self):
    action = 'example:early_and_fail'
    self.assertRaises(exception.ForbiddenAction, policy.enforce, self.credentials, action, self.target)
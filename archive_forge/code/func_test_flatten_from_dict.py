import copy
from unittest import mock
from oslo_serialization import jsonutils
from oslo_policy import shell
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_flatten_from_dict(self):
    target = {'target': {'secret': {'project_id': '1234'}}}
    result = shell.flatten(target)
    self.assertEqual(result, {'target.secret.project_id': '1234'})
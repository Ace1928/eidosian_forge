import unittest
import six
from apitools.base.py.testing import mock
from samples.iam_sample.iam_v1 import iam_v1_client  # nopep8
from samples.iam_sample.iam_v1 import iam_v1_messages  # nopep8
def testFlatPath(self):
    get_method_config = self.mocked_iam_v1.projects_serviceAccounts_keys.GetMethodConfig('Get')
    self.assertEquals('v1/projects/{projectsId}/serviceAccounts/{serviceAccountsId}/keys/{keysId}', get_method_config.flat_path)
    self.assertEquals('v1/{+name}', get_method_config.relative_path)
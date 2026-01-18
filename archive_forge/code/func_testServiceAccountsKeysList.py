import unittest
import six
from apitools.base.py.testing import mock
from samples.iam_sample.iam_v1 import iam_v1_client  # nopep8
from samples.iam_sample.iam_v1 import iam_v1_messages  # nopep8
def testServiceAccountsKeysList(self):
    response_key = iam_v1_messages.ServiceAccountKey(name=u'test-key')
    self.mocked_iam_v1.projects_serviceAccounts_keys.List.Expect(iam_v1_messages.IamProjectsServiceAccountsKeysListRequest(name=u'test-service-account.'), iam_v1_messages.ListServiceAccountKeysResponse(keys=[response_key]))
    result = self.mocked_iam_v1.projects_serviceAccounts_keys.List(iam_v1_messages.IamProjectsServiceAccountsKeysListRequest(name=u'test-service-account.'))
    self.assertEquals([response_key], result.keys)
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from random import randint
from unittest import mock
from gslib.cloud_api import AccessDeniedException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
@mock.patch('gslib.cloud_api_delegator.CloudApiDelegator.GetProjectServiceAccount')
@mock.patch('gslib.cloud_api_delegator.CloudApiDelegator.PatchBucket')
@mock.patch('gslib.kms_api.KmsApi.GetKeyIamPolicy')
@mock.patch('gslib.kms_api.KmsApi.SetKeyIamPolicy')
def testEncryptionSetKeyFailsWhenUpdateKeyPolicyFailsWithoutWarningFlag(self, mock_set_key_iam_policy, mock_get_key_iam_policy, mock_patch_bucket, mock_get_project_service_account):
    bucket_uri = self.CreateBucket()
    mock_get_key_iam_policy.side_effect = AccessDeniedException('Permission denied')
    mock_get_project_service_account.return_value.email_address = 'dummy@google.com'
    try:
        stdout = self.RunCommand('kms', ['encryption', '-k', self.dummy_keyname, suri(bucket_uri)], return_stdout=True)
        self.fail('Did not get expected AccessDeniedException')
    except AccessDeniedException as e:
        self.assertIn('Permission denied', e.reason)
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import boto
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from gslib.cloud_api import AccessDeniedException
from gslib.cred_types import CredTypes
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.tests.mock_logging_handler import MockLoggingHandler
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from six import add_move, MovedModule
from six.moves import mock
from datetime import datetime
@SkipForS3('Tests only uses gs credentials.')
@SkipForXML('Tests only run on JSON API.')
@mock.patch('gslib.third_party.iamcredentials_apitools.iamcredentials_v1_client.IamcredentialsV1.ProjectsServiceAccountsService.GenerateAccessToken')
def testImpersonationBlockedByIamCredentialsApiErrors(self, mock_iam_creds_generate_access_token):
    with SetBotoConfigForTest([('Credentials', 'gs_oauth2_refresh_token', 'foo'), ('Credentials', 'gs_service_client_id', None), ('Credentials', 'gs_service_key_file', None), ('Credentials', 'gs_impersonate_service_account', 'bar')]):
        mock_iam_creds_generate_access_token.side_effect = _Make403('The caller does not have permission')
        try:
            GcsJsonApi(None, self.logger, DiscardMessagesQueue())
            self.fail('Succeeded when IAM Credentials threw an error')
        except AccessDeniedException as e:
            self.assertIn('Service account impersonation failed.', str(e))
        mock_iam_creds_generate_access_token.side_effect = _Make403('IAM Service Account Credentials API has not been used')
        try:
            GcsJsonApi(None, self.logger, DiscardMessagesQueue())
            self.fail('Succeeded when IAM Credentials threw an error')
        except AccessDeniedException as e:
            self.assertIn('IAM Service Account Credentials API has not', str(e))
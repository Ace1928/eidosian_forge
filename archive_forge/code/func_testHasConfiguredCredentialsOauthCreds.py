from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import boto.auth
from gslib import cloud_api
from gslib.utils import boto_util
from gslib import context_config
from gslib.tests import testcase
from gslib.tests.testcase import base
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from six import add_move, MovedModule
from six.moves import mock
@mock.patch.object(boto.auth, 'get_auth_handler', return_value=None)
def testHasConfiguredCredentialsOauthCreds(self, _):
    with SetBotoConfigForTest([('Credentials', 'gs_access_key_id', None), ('Credentials', 'gs_secret_access_key', None), ('Credentials', 'aws_access_key_id', None), ('Credentials', 'aws_secret_access_key', None), ('Credentials', 'gs_oauth2_refresh_token', '?????'), ('Credentials', 'gs_external_account_file', None), ('Credentials', 'gs_external_account_authorized_user_file', None), ('Credentials', 'gs_service_client_id', None), ('Credentials', 'gs_service_key_file', None)]):
        self.assertTrue(boto_util.HasConfiguredCredentials())
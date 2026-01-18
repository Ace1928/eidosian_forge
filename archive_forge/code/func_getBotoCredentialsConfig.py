from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import GceAssertionCredentials
from google_reauth import reauth_creds
from gslib import gcs_json_api
from gslib import gcs_json_credentials
from gslib.cred_types import CredTypes
from gslib.exception import CommandException
from gslib.tests import testcase
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.utils.wrapped_credentials import WrappedCredentials
import logging
from oauth2client.service_account import ServiceAccountCredentials
import pkgutil
from six import add_move, MovedModule
from six.moves import mock
def getBotoCredentialsConfig(service_account_creds=None, user_account_creds=None, gce_creds=None, external_account_creds=None, external_account_authorized_user_creds=None):
    config = []
    if service_account_creds:
        config.append(('Credentials', 'gs_service_key_file', service_account_creds['keyfile']))
        config.append(('Credentials', 'gs_service_client_id', service_account_creds['client_id']))
    else:
        config.append(('Credentials', 'gs_service_key_file', None))
    config.extend([('Credentials', 'gs_oauth2_refresh_token', user_account_creds), ('GoogleCompute', 'service_account', gce_creds), ('Credentials', 'gs_external_account_file', external_account_creds), ('Credentials', 'gs_external_account_authorized_user_file', external_account_authorized_user_creds)])
    return config
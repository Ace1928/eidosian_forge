import os
import sys
import time
import random
import urllib
import datetime
import unittest
import threading
from unittest import mock
import requests
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.google import (
def test_auth_connection(self):
    self.assertRaises(GoogleAuthError, GoogleOAuth2Credential, *GCE_PARAMS, **{'auth_type': 'XX'})
    self.assertRaises(GoogleAuthError, GoogleOAuth2Credential, *GCE_PARAMS, **{'auth_type': GoogleAuthType.GCS_S3})
    kwargs = {}
    if SHA256:
        kwargs['auth_type'] = GoogleAuthType.SA
        cred1 = GoogleOAuth2Credential(*GCE_PARAMS_PEM_KEY_FILE, **kwargs)
        self.assertTrue(isinstance(cred1.oauth2_conn, GoogleServiceAcctAuthConnection))
        cred1 = GoogleOAuth2Credential(*GCE_PARAMS_JSON_KEY_FILE, **kwargs)
        self.assertTrue(isinstance(cred1.oauth2_conn, GoogleServiceAcctAuthConnection))
        cred1 = GoogleOAuth2Credential(*GCE_PARAMS_PEM_KEY, **kwargs)
        self.assertTrue(isinstance(cred1.oauth2_conn, GoogleServiceAcctAuthConnection))
        cred1 = GoogleOAuth2Credential(*GCE_PARAMS_JSON_KEY, **kwargs)
        self.assertTrue(isinstance(cred1.oauth2_conn, GoogleServiceAcctAuthConnection))
        cred1 = GoogleOAuth2Credential(*GCE_PARAMS_KEY, **kwargs)
        self.assertTrue(isinstance(cred1.oauth2_conn, GoogleServiceAcctAuthConnection))
        kwargs['auth_type'] = GoogleAuthType.SA
        cred1 = GoogleOAuth2Credential(*GCE_PARAMS_JSON_KEY_STR, **kwargs)
        self.assertTrue(isinstance(cred1.oauth2_conn, GoogleServiceAcctAuthConnection))
        self.assertRaises(GoogleAuthError, GoogleOAuth2Credential, *GCE_PARAMS, **kwargs)
        kwargs['auth_type'] = GoogleAuthType.SA
        expected_msg = 'Unable to decode provided PEM key:'
        self.assertRaisesRegex(GoogleAuthError, expected_msg, GoogleOAuth2Credential, *GCE_PARAMS_PEM_KEY_FILE_INVALID, **kwargs)
        kwargs['auth_type'] = GoogleAuthType.SA
        expected_msg = 'Unable to decode provided PEM key:'
        self.assertRaisesRegex(GoogleAuthError, expected_msg, GoogleOAuth2Credential, *GCE_PARAMS_JSON_KEY_INVALID, **kwargs)
    kwargs['auth_type'] = GoogleAuthType.IA
    cred2 = GoogleOAuth2Credential(*GCE_PARAMS_IA, **kwargs)
    self.assertTrue(isinstance(cred2.oauth2_conn, GoogleInstalledAppAuthConnection))
    kwargs['auth_type'] = GoogleAuthType.GCE
    cred3 = GoogleOAuth2Credential(*GCE_PARAMS_GCE, **kwargs)
    self.assertTrue(isinstance(cred3.oauth2_conn, GoogleGCEServiceAcctAuthConnection))
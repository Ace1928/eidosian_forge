from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_keyring_is_used(self):
    self.config = {'Credentials': {'aws_access_key_id': 'cfg_access_key', 'keyring': 'test'}}
    import sys
    try:
        import keyring
        imported = True
    except ImportError:
        sys.modules['keyring'] = keyring = type(mock)('keyring', '')
        imported = False
    try:
        with mock.patch('keyring.get_password', create=True):
            keyring.get_password.side_effect = lambda kr, login: kr + login + 'pw'
            p = provider.Provider('aws')
            self.assertEqual(p.access_key, 'cfg_access_key')
            self.assertEqual(p.secret_key, 'testcfg_access_keypw')
            self.assertIsNone(p.security_token)
    finally:
        if not imported:
            del sys.modules['keyring']
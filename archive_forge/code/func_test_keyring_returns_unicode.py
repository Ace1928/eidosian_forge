import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
def test_keyring_returns_unicode(self):

    class UnicodeInMemoryKeyring(InMemoryKeyring):

        def get_password(self, service, username):
            password = super(UnicodeInMemoryKeyring, self).get_password(service, username)
            if isinstance(password, unicode_type):
                password = password.encode('utf-8')
            return password
    self.keyring = UnicodeInMemoryKeyring()
    with fake_keyring(self.keyring):
        credential = self.make_credential('consumer key')
        self.assertTrue(credential)
        self.store.save(credential, 'unique key')
        credential2 = self.store.load('unique key')
        self.assertTrue(credential2)
        self.assertEqual(credential.consumer.key, credential2.consumer.key)
        self.assertEqual(credential.consumer.secret, credential2.consumer.secret)
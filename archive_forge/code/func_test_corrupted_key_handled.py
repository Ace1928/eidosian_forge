import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
def test_corrupted_key_handled(self):

    class CorruptedInMemoryKeyring(InMemoryKeyring):

        def get_password(self, service, username):
            return 'bad'
    self.keyring = CorruptedInMemoryKeyring()
    with fake_keyring(self.keyring):
        credential = self.make_credential('consumer key')
        self.store.save(credential, 'unique key')
        credential2 = self.store.load('unique key')
        self.assertIsNone(credential2)
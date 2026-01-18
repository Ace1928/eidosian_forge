import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
def test_lookup_by_unique_key(self):
    with fake_keyring(self.keyring):
        credential1 = self.make_credential('consumer key1')
        self.store.save(credential1, 'key 1')
        credential2 = self.make_credential('consumer key2')
        self.store.save(credential2, 'key 2')
        loaded1 = self.store.load('key 1')
        self.assertTrue(loaded1)
        self.assertEqual(credential1.consumer.key, loaded1.consumer.key)
        loaded2 = self.store.load('key 2')
        self.assertEqual(credential2.consumer.key, loaded2.consumer.key)
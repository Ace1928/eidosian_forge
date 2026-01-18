import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
def test_bad_unique_id_returns_none(self):
    with fake_keyring(self.keyring):
        self.assertIsNone(self.store.load('no such key'))
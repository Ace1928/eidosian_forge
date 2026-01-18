from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
def test_get_vault(self):
    self.mock_layer1.describe_vault.return_value = FIXTURE_VAULT
    vault = self.layer2.get_vault('examplevault')
    self.assertEqual(vault.layer1, self.mock_layer1)
    self.assertEqual(vault.name, 'examplevault')
    self.assertEqual(vault.size, 78088912)
    self.assertEqual(vault.number_of_archives, 192)
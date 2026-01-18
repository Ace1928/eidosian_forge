from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
def test_list_vaults_paginated(self):
    resps = [FIXTURE_PAGINATED_VAULTS, FIXTURE_PAGINATED_VAULTS_CONT]

    def return_paginated_vaults_resp(marker=None, limit=None):
        return resps.pop(0)
    self.mock_layer1.list_vaults = Mock(side_effect=return_paginated_vaults_resp)
    vaults = self.layer2.list_vaults()
    self.assertEqual(vaults[0].name, 'vault0')
    self.assertEqual(vaults[3].name, 'vault3')
    self.assertEqual(len(vaults), 4)
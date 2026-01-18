from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
def test_list_all_parts_one_page(self):
    self.mock_layer1.list_parts.return_value = dict(EXAMPLE_PART_LIST_COMPLETE)
    parts_result = self.vault.list_all_parts(sentinel.upload_id)
    expected = [call('examplevault', sentinel.upload_id)]
    self.assertEquals(expected, self.mock_layer1.list_parts.call_args_list)
    self.assertEquals(EXAMPLE_PART_LIST_COMPLETE, parts_result)
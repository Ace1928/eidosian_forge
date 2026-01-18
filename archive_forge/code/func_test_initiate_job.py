from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
def test_initiate_job(self):

    class UTC(tzinfo):
        """UTC"""

        def utcoffset(self, dt):
            return timedelta(0)

        def tzname(self, dt):
            return 'Z'

        def dst(self, dt):
            return timedelta(0)
    self.mock_layer1.initiate_job.return_value = {'JobId': 'job-id'}
    self.vault.retrieve_inventory(start_date=datetime(2014, 1, 1, tzinfo=UTC()), end_date=datetime(2014, 1, 2, tzinfo=UTC()), limit=100)
    self.mock_layer1.initiate_job.assert_called_with('examplevault', {'Type': 'inventory-retrieval', 'InventoryRetrievalParameters': {'StartDate': '2014-01-01T00:00:00Z', 'EndDate': '2014-01-02T00:00:00Z', 'Limit': 100}})
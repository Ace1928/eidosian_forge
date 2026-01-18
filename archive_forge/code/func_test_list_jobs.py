from tests.unit import unittest
from mock import call, Mock, patch, sentinel
import codecs
from boto.glacier.layer1 import Layer1
from boto.glacier.layer2 import Layer2
import boto.glacier.vault
from boto.glacier.vault import Vault
from boto.glacier.vault import Job
from datetime import datetime, tzinfo, timedelta
def test_list_jobs(self):
    self.mock_layer1.list_jobs.return_value = {'JobList': [FIXTURE_ARCHIVE_JOB]}
    jobs = self.vault.list_jobs(False, 'InProgress')
    self.mock_layer1.list_jobs.assert_called_with('examplevault', False, 'InProgress')
    self.assertEqual(jobs[0].archive_id, 'NkbByEejwEggmBz2fTHgJrg0XBoDfjP4q6iu87-TjhqG6eGoOY9Z8i1_AUyUsuhPAdTqLHy8pTl5nfCFJmDl2yEZONi5L26Omw12vcs01MNGntHEQL8MBfGlqrEXAMPLEArchiveId')
import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_commit_put_no_retry_on_conflict(self):
    self.session.retriable_status_codes = {409, 503}
    self._test_commit(commit_method='PATCH', commit_args={'retry_on_conflict': False}, expected_args={'retriable_status_codes': {503}})
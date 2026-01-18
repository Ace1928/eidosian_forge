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
def test_commit_put_explicit_microversion(self):
    self._test_commit(commit_method='PUT', prepend_key=True, has_body=True, explicit_microversion='1.42')
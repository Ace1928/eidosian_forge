import collections
import os
import tempfile
import time
import urllib
import uuid
import fixtures
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from requests import structures
from requests_mock.contrib import fixture as rm_fixture
import openstack.cloud
import openstack.config as occ
import openstack.connection
from openstack.fixture import connection as os_fixture
from openstack.tests import base
from openstack.tests import fakes
def get_nova_discovery_mock_dict(self, compute_version_json='compute-version.json', compute_discovery_url='https://compute.example.com/v2.1/'):
    discovery_fixture = os.path.join(self.fixtures_directory, compute_version_json)
    return dict(method='GET', uri=compute_discovery_url, text=open(discovery_fixture, 'r').read())
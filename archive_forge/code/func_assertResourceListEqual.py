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
def assertResourceListEqual(self, actual, expected, resource_type):
    """Helper for the assertEqual which compares Resource lists object
        against dictionary representing expected state.

        :param list actual: List of actual objects.
        :param listexpected: List of dictionaries representing expected
            objects.
        :param class resource_type: class type to be applied for the expected
            resource.
        """
    self.assertEqual([resource_type(**f).to_dict(computed=False) for f in expected], [f.to_dict(computed=False) for f in actual])
import os
from unittest import mock
import fixtures
from keystoneauth1 import session
from testtools import matchers
import openstack.config
from openstack import connection
from openstack import proxy
from openstack import service_description
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack.tests.unit.fake import fake_service
def test_create_connection_version_param_bogus(self):
    c1 = connection.Connection(cloud='sample-cloud')
    conn = connection.Connection(session=c1.session, identity_api_version='red')
    self.assertEqual('openstack.identity.v3._proxy', conn.identity.__class__.__module__)
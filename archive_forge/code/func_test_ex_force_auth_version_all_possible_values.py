import os
import sys
import datetime
import unittest
from unittest import mock
from unittest.mock import Mock, patch
import pytest
import requests_mock
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.pricing import set_pricing, clear_pricing_data
from libcloud.utils.py3 import u, httplib, method_type
from libcloud.common.base import LibcloudConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import OpenStackFixtures, ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import (
def test_ex_force_auth_version_all_possible_values(self):
    """
        Test case which verifies that the driver can be correctly instantiated using all the
        supported API versions.
        """
    cls = get_driver(Provider.OPENSTACK)
    for auth_version in AUTH_VERSIONS_WITH_EXPIRES:
        driver_kwargs = {}
        if auth_version in ['1.1', '3.0']:
            continue
        user_id = OPENSTACK_PARAMS[0]
        key = OPENSTACK_PARAMS[1]
        if auth_version.startswith('3.x'):
            driver_kwargs['ex_domain_name'] = 'test_domain'
            driver_kwargs['ex_tenant_domain_id'] = 'test_tenant_domain_id'
            driver_kwargs['ex_force_service_region'] = 'regionOne'
            driver_kwargs['ex_tenant_name'] = 'tenant-name'
        if auth_version == '3.x_oidc_access_token':
            key = 'test_key'
            driver_kwargs['ex_domain_name'] = None
        elif auth_version == '3.x_appcred':
            user_id = 'appcred_id'
            key = 'appcred_secret'
        driver = cls(user_id, key, ex_force_auth_url='http://x.y.z.y:5000', ex_force_auth_version=auth_version, **driver_kwargs)
        nodes = driver.list_nodes()
        self.assertTrue(len(nodes) >= 1)
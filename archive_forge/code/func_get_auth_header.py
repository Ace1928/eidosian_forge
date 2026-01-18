import hashlib
import http.client as http
import os
import subprocess
import tempfile
import time
import urllib
import uuid
import fixtures
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_serialization import jsonutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional import ft_utils as func_utils
from glance.tests import utils as test_utils
def get_auth_header(tenant, tenant_id=None, role='reader,member', headers=None):
    """Return headers to authenticate as a specific tenant.

    :param tenant: Tenant for the auth token
    :param tenant_id: Optional tenant ID for the X-Tenant-Id header
    :param role: Optional user role
    :param headers: Optional list of headers to add to
    """
    if not headers:
        headers = {}
    auth_token = 'user:%s:%s' % (tenant, role)
    headers.update({'X-Auth-Token': auth_token})
    if tenant_id:
        headers.update({'X-Tenant-Id': tenant_id})
    return headers
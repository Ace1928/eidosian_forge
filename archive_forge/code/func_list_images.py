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
def list_images(tenant, role='', visibility=None):
    auth_token = 'user:%s:%s' % (tenant, role)
    headers = {'X-Auth-Token': auth_token}
    path = self._url('/v2/images')
    if visibility is not None:
        path += '?visibility=%s' % visibility
    response = requests.get(path, headers=headers)
    self.assertEqual(http.OK, response.status_code)
    return jsonutils.loads(response.text)['images']
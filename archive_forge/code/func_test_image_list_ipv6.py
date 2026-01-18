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
def test_image_list_ipv6(self):
    self.api_server.deployment_flavor = 'caching'
    self.start_servers(**self.__dict__.copy())
    url = f'http://[::1]:{self.api_port}'
    path = '/'
    requests.get(url + path, headers=self._headers())
    path = '/v2/images'
    response = requests.get(url + path, headers=self._headers())
    self.assertEqual(200, response.status_code)
    images = jsonutils.loads(response.text)['images']
    self.assertEqual(0, len(images))
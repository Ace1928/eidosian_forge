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
def test_image_visibility_to_different_users(self):
    self.cleanup()
    self.api_server.deployment_flavor = 'fakeauth'
    kwargs = self.__dict__.copy()
    self.start_servers(**kwargs)
    owners = ['admin', 'tenant1', 'tenant2', 'none']
    visibilities = ['public', 'private', 'shared', 'community']
    for owner in owners:
        for visibility in visibilities:
            path = self._url('/v2/images')
            headers = self._headers({'content-type': 'application/json', 'X-Auth-Token': 'createuser:%s:admin' % owner})
            data = jsonutils.dumps({'name': '%s-%s' % (owner, visibility), 'visibility': visibility})
            response = requests.post(path, headers=headers, data=data)
            self.assertEqual(http.CREATED, response.status_code)

    def list_images(tenant, role='', visibility=None):
        auth_token = 'user:%s:%s' % (tenant, role)
        headers = {'X-Auth-Token': auth_token}
        path = self._url('/v2/images')
        if visibility is not None:
            path += '?visibility=%s' % visibility
        response = requests.get(path, headers=headers)
        self.assertEqual(http.OK, response.status_code)
        return jsonutils.loads(response.text)['images']
    images = list_images('tenant1', role='reader')
    self.assertEqual(7, len(images))
    for image in images:
        self.assertTrue(image['visibility'] == 'public' or 'tenant1' in image['name'])
    images = list_images('tenant1', role='reader', visibility='public')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('public', image['visibility'])
    images = list_images('tenant1', role='reader', visibility='private')
    self.assertEqual(1, len(images))
    image = images[0]
    self.assertEqual('private', image['visibility'])
    self.assertIn('tenant1', image['name'])
    images = list_images('tenant1', role='reader', visibility='shared')
    self.assertEqual(1, len(images))
    image = images[0]
    self.assertEqual('shared', image['visibility'])
    self.assertIn('tenant1', image['name'])
    images = list_images('tenant1', role='reader', visibility='community')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('community', image['visibility'])
    images = list_images('none', role='reader')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('public', image['visibility'])
    images = list_images('none', role='reader', visibility='public')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('public', image['visibility'])
    images = list_images('none', role='reader', visibility='private')
    self.assertEqual(0, len(images))
    images = list_images('none', role='reader', visibility='shared')
    self.assertEqual(0, len(images))
    images = list_images('none', role='reader', visibility='community')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('community', image['visibility'])
    images = list_images('none', role='admin')
    self.assertEqual(12, len(images))
    images = list_images('none', role='admin', visibility='public')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('public', image['visibility'])
    images = list_images('none', role='admin', visibility='private')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('private', image['visibility'])
    images = list_images('none', role='admin', visibility='shared')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('shared', image['visibility'])
    images = list_images('none', role='admin', visibility='community')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('community', image['visibility'])
    images = list_images('admin', role='admin')
    self.assertEqual(13, len(images))
    images = list_images('admin', role='admin', visibility='public')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('public', image['visibility'])
    images = list_images('admin', role='admin', visibility='private')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('private', image['visibility'])
    images = list_images('admin', role='admin', visibility='shared')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('shared', image['visibility'])
    images = list_images('admin', role='admin', visibility='community')
    self.assertEqual(4, len(images))
    for image in images:
        self.assertEqual('community', image['visibility'])
    self.stop_servers()
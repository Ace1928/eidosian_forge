import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
def test_headers_are_unicode(self):
    """
        Verifies that the headers returned by conversion code are unicode.

        Headers are passed via http in non-testing mode, which automatically
        converts them to unicode. Verifying that the method does the
        conversion proves that we aren't passing data that works in tests
        but will fail in production.
        """
    fixture = {'name': 'fake public image', 'is_public': True, 'size': 19, 'location': 'file:///tmp/glance-tests/2', 'properties': {'distro': 'Ubuntu 10.04 LTS'}}
    headers = utils.image_meta_to_http_headers(fixture)
    for k, v in headers.items():
        self.assertIsInstance(v, str)
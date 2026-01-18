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
def test_get_action_args_del_format_error(self):
    actions = {'action': 'update', 'id': 12}
    env = {'wsgiorg.routing_args': [None, actions]}
    expected = {'action': 'update', 'id': 12}
    actual = wsgi.Resource(None, None, None).get_action_args(env)
    self.assertEqual(expected, actual)
from unittest import mock
import fixtures
import json
from oslo_config import cfg
import socket
import webob
from heat.api.aws import exception as aws_exception
from heat.common import exception
from heat.common import wsgi
from heat.tests import common
def test_best_match_language(self):
    request = wsgi.Request.blank('/')
    accepted = 'unknown-lang'
    request.headers = {'Accept-Language': accepted}

    def fake_best_match(self, offers, default_match=None):
        return None
    with mock.patch.object(request.accept_language, 'best_match') as mock_match:
        mock_match.side_effect = fake_best_match
    self.assertIsNone(request.best_match_language())
    request.headers = {'Accept-Language': ''}
    self.assertIsNone(request.best_match_language())
    request.headers.pop('Accept-Language')
    self.assertIsNone(request.best_match_language())
import argparse
from io import StringIO
import itertools
import logging
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from oslo_serialization import jsonutils
import requests
from testtools import matchers
from keystoneclient import adapter
from keystoneclient.auth import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient import session as client_session
from keystoneclient.tests.unit import utils
def test_connect_retries(self):

    def _timeout_error(request, context):
        raise requests.exceptions.Timeout()
    self.stub_url('GET', text=_timeout_error)
    session = client_session.Session()
    retries = 3
    with mock.patch('time.sleep') as m:
        self.assertRaises(exceptions.RequestTimeout, session.get, self.TEST_URL, connect_retries=retries)
        self.assertEqual(retries, m.call_count)
        m.assert_called_with(2.0)
    self.assertThat(self.requests_mock.request_history, matchers.HasLength(retries + 1))
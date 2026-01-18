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
def test_mask_password_in_http_log_response(self):
    session = client_session.Session()

    def fake_debug(msg):
        self.assertNotIn('verybadpass', msg)
    logger = mock.Mock(isEnabledFor=mock.Mock(return_value=True))
    logger.debug = mock.Mock(side_effect=fake_debug)
    body = {'connection_info': {'driver_volume_type': 'iscsi', 'data': {'auth_password': 'verybadpass', 'target_discovered': False, 'encrypted': False, 'qos_specs': None, 'target_iqn': 'iqn.2010-10.org.openstack:volume-744d2085-8e78-40a5-8659-ef3cffb2480e', 'target_portal': '172.99.69.228:3260', 'volume_id': '744d2085-8e78-40a5-8659-ef3cffb2480e', 'target_lun': 1, 'access_mode': 'rw', 'auth_username': 'verybadusername', 'auth_method': 'CHAP'}}}
    body_json = jsonutils.dumps(body)
    response = mock.Mock(text=body_json, status_code=200, headers={'content-type': 'application/json'})
    session._http_log_response(response, logger)
    self.assertEqual(1, logger.debug.call_count)
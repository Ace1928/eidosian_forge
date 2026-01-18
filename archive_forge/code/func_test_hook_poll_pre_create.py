import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
def test_hook_poll_pre_create(self):
    self.register_keystone_auth_fixture()
    stack_id = 'teststack/1'
    nested_id = 'nested/2'
    self._stub_responses(stack_id, nested_id, 'CREATE')
    list_text = self.shell('hook-poll %s --nested-depth 1' % stack_id)
    hook_reason = 'CREATE paused until Hook pre-create is cleared'
    required = ['id', 'p_eventid2', 'stack_name', 'teststack', hook_reason]
    for r in required:
        self.assertRegex(list_text, r)
    self.assertNotRegex(list_text, 'p_eventid1')
    self.assertNotRegex(list_text, 'n_eventid1')
    self.assertNotRegex(list_text, 'n_eventid2')
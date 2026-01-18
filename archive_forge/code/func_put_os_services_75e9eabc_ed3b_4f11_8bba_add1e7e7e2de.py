import copy
import datetime
import re
from unittest import mock
from urllib import parse
from oslo_utils import strutils
import novaclient
from novaclient import api_versions
from novaclient import client as base_client
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils
from novaclient.v2 import client
def put_os_services_75e9eabc_ed3b_4f11_8bba_add1e7e7e2de(self, body, **kw):
    """This should only be called with microversion >= 2.53."""
    return (200, FAKE_RESPONSE_HEADERS, {'service': {'host': 'host1', 'binary': 'nova-compute', 'status': body.get('status', 'enabled'), 'disabled_reason': body.get('disabled_reason'), 'forced_down': body.get('forced_down', False)}})
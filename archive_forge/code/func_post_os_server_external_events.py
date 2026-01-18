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
def post_os_server_external_events(self, **kw):
    return (200, FAKE_RESPONSE_HEADERS, {'events': [{'name': 'test-event', 'status': 'completed', 'tag': 'tag', 'server_uuid': 'fake-uuid1'}, {'name': 'test-event', 'status': 'completed', 'tag': 'tag', 'server_uuid': 'fake-uuid2'}]})
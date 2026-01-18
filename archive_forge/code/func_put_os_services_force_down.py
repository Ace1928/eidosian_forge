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
def put_os_services_force_down(self, body, **kw):
    return (200, FAKE_RESPONSE_HEADERS, {'service': {'host': body['host'], 'binary': body['binary'], 'forced_down': False}})
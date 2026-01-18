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
def get_flavors_2_os_flavor_access(self, **kw):
    return (200, FAKE_RESPONSE_HEADERS, {'flavor_access': [{'flavor_id': '2', 'tenant_id': 'proj1'}, {'flavor_id': '2', 'tenant_id': 'proj2'}]})
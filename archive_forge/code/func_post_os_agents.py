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
def post_os_agents(self, body):
    return (200, {}, {'agent': {'url': '/xxx/xxx/xxx', 'hypervisor': body['agent']['hypervisor'], 'md5hash': 'add6bb58e139be103324d04d82d8f546', 'version': '7.0', 'architecture': 'x86', 'os': 'win', 'id': 1}})
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
def get_os_hypervisors_region_child_1_uptime(self, **kw):
    return (200, {}, {'hypervisor': {'id': 'region!child@1', 'hypervisor_hostname': 'hyper1', 'uptime': 'fake uptime'}})
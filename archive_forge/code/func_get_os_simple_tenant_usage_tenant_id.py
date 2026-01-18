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
def get_os_simple_tenant_usage_tenant_id(self, **kw):
    return (200, {}, {'tenant_usage': {'total_memory_mb_usage': 25451.762807466665, 'total_vcpus_usage': 49.71047423333333, 'total_hours': 49.71047423333333, 'tenant_id': '7b0a1d73f8fb41718f3343c207597869', 'stop': '2012-01-22 19:48:41.750722', 'server_usages': [{'hours': 49.71047423333333, 'uptime': 27035, 'local_gb': 0, 'ended_at': None, 'name': 'f15image1', 'tenant_id': '7b0a1d73f8fb41718f3343c207597869', 'instance_id': 'f079e394-1111-457b-b350-bb5ecc685cdd', 'vcpus': 1, 'memory_mb': 512, 'state': 'active', 'flavor': 'm1.tiny', 'started_at': '2012-01-20 18:06:06.479998'}], 'start': '2011-12-25 19:48:41.750687', 'total_local_gb_usage': 0.0}})
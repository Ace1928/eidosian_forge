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
def get_v2_0_networks(self, **kw):
    """Return neutron proxied networks.

        We establish a few different possible networks that we can get
        by name, which we can then call in tests. The only usage of
        this API should be for name -> id translation, however a full
        valid neutron block is provided for the private network to see
        the kinds of things that will be in that payload.
        """
    name = kw.get('name', 'blank')
    networks_by_name = {'private': [{'status': 'ACTIVE', 'router:external': False, 'availability_zone_hints': [], 'availability_zones': ['nova'], 'description': '', 'name': 'private', 'subnets': ['64706c26-336c-4048-ab3c-23e3283bca2c', '18512740-c760-4d5f-921f-668105c9ee44'], 'shared': False, 'tenant_id': 'abd42f270bca43ea80fe4a166bc3536c', 'created_at': '2016-08-15T17:34:49', 'tags': [], 'ipv6_address_scope': None, 'updated_at': '2016-08-15T17:34:49', 'admin_state_up': True, 'ipv4_address_scope': None, 'port_security_enabled': True, 'mtu': 1450, 'id': 'e43a56c7-11d4-45c9-8681-ddc8171b5850', 'revision': 2}], 'duplicate': [{'status': 'ACTIVE', 'id': 'e43a56c7-11d4-45c9-8681-ddc8171b5850'}, {'status': 'ACTIVE', 'id': 'f43a56c7-11d4-45c9-8681-ddc8171b5850'}], 'blank': []}
    return (200, {}, {'networks': networks_by_name[name]})
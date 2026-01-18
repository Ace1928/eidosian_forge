import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_build_userdata_with_invalid_ignition(self):
    metadata = {'os-collect-config': {'heat': {'password': '***'}}}
    userdata = '{"ignition": {"version": "3.0"}, "storage": []}'
    ud_format = 'SOFTWARE_CONFIG'
    self.assertRaises(ValueError, self.nova_plugin.build_userdata, metadata, userdata=userdata, user_data_format=ud_format)
import io
import sys
from unittest import mock
from urllib import parse
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils as test_utils
from novaclient import utils
from novaclient.v2 import servers
def test_do_action_on_many_resource_string_with_human_id(self):
    resource = servers.Server(None, {'name': 'resource1'})
    self._test_do_action_on_many_resource_string(resource, 'resource1')
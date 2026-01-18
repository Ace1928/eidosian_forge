import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_get_ip_address(self):
    self.assertEqual('192.168.0.3', server._get_ip_address(self.OLD, 'private', [4, 6]))
    self.assertEqual('10.10.1.2', server._get_ip_address(self.NEW, 'fixed', [4, 6]))
    self.assertEqual('10.10.1.2', server._get_ip_address(self.NEW, 'private', [4, 6]))
    self.assertEqual('0:0:0:0:0:ffff:a0a:103', server._get_ip_address(self.NEW, 'public', [6]))
    self.assertEqual('0:0:0:0:0:ffff:a0a:103', server._get_ip_address(self.NEW, 'floating', [6]))
    self.assertEqual('124.12.125.4', server._get_ip_address(self.ODD, 'public', [4, 6]))
    self.assertEqual('10.3.3.18', server._get_ip_address(self.ODD, 'private', [4, 6]))
    self.assertRaises(exceptions.CommandError, server._get_ip_address, self.NEW, 'public', [4])
    self.assertRaises(exceptions.CommandError, server._get_ip_address, self.NEW, 'admin', [4])
    self.assertRaises(exceptions.CommandError, server._get_ip_address, self.OLD, 'public', [4, 6])
    self.assertRaises(exceptions.CommandError, server._get_ip_address, self.OLD, 'private', [6])
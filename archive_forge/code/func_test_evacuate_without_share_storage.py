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
def test_evacuate_without_share_storage(self):
    self.compute_client.api_version = api_versions.APIVersion('2.13')
    args = [self.server.id, '--shared-storage']
    verify_args = [('server', self.server.id), ('shared_storage', True)]
    evac_args = {'host': None, 'on_shared_storage': True, 'password': None}
    self._test_evacuate(args, verify_args, evac_args)
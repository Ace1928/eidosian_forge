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
def datalist(self):
    datalist = (server.PowerStateColumn(getattr(self.new_server, 'OS-EXT-STS:power_state')), format_columns.DictListColumn({}), self.flavor.name + ' (' + self.new_server.flavor.get('id') + ')', self.new_server.id, self.image.name + ' (' + self.new_server.image.get('id') + ')', self.new_server.name, format_columns.DictColumn(self.new_server.metadata))
    return datalist
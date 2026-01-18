import itertools
from unittest import mock
import ddt
import fixtures
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient import client
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils as test_utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient import utils
from manilaclient.v2 import messages
from manilaclient.v2 import security_services
from manilaclient.v2 import share_access_rules
from manilaclient.v2 import share_group_types
from manilaclient.v2 import share_groups
from manilaclient.v2 import share_instances
from manilaclient.v2 import share_network_subnets
from manilaclient.v2 import share_networks
from manilaclient.v2 import share_servers
from manilaclient.v2 import share_snapshots
from manilaclient.v2 import share_types
from manilaclient.v2 import shares
from manilaclient.v2 import shell as shell_v2
def test_extract_metadata(self):

    class Arguments(object):

        def __init__(self, metadata=None):
            if metadata is None:
                metadata = []
            self.metadata = metadata
    inputs = [([], {}), (['key=value'], {'key': 'value'}), (['key'], {'key': None}), (['k1=v1', 'k2=v2'], {'k1': 'v1', 'k2': 'v2'}), (['k1=v1', 'k2'], {'k1': 'v1', 'k2': None}), (['k1', 'k2=v2'], {'k1': None, 'k2': 'v2'})]
    for input in inputs:
        args = Arguments(metadata=input[0])
        self.assertEqual(shell_v2._extract_metadata(args), input[1])
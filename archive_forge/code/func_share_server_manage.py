import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def share_server_manage(self, host, share_network, identifier, driver_options=None):
    if driver_options:
        command = 'share-server-manage %s %s %s %s' % (host, share_network, identifier, driver_options)
    else:
        command = 'share-server-manage %s %s %s' % (host, share_network, identifier)
    managed_share_server_raw = self.manila(command)
    managed_share_server = output_parser.details(managed_share_server_raw)
    return managed_share_server['id']
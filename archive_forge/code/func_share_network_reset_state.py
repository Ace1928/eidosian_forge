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
def share_network_reset_state(self, id=None, state=None, microversion=None):
    cmd = 'share-network-reset-state %s ' % id
    if state:
        cmd += '--state %s' % state
    share_network_raw = self.manila(cmd, microversion=microversion)
    share_network = utils.listing(share_network_raw)
    return share_network
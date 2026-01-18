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
def reset_snapshot_instance(self, id=None, state=None, microversion=None):
    """Reset snapshot instance status."""
    cmd = 'snapshot-instance-reset-state %s ' % id
    if state:
        cmd += '--state %s' % state
    snapshot_instance_raw = self.manila(cmd, microversion=microversion)
    snapshot_instance = utils.listing(snapshot_instance_raw)
    return snapshot_instance
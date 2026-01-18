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
def get_share_transfer(self, transfer, microversion=None):
    """Get a share transfer.

        :param transfer: ID or name of share transfer.
        """
    cmd = 'share-transfer-show %s ' % transfer
    transfer_raw = self.manila(cmd, microversion=microversion)
    transfer = output_parser.details(transfer_raw)
    return transfer
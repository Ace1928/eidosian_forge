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
def is_snapshot_deleted(self, snapshot, microversion=None):
    """Indicates whether snapshot is deleted or not.

        :param snapshot: str -- Name or ID of snapshot
        """
    try:
        self.get_snapshot(snapshot, microversion=microversion)
        return False
    except tempest_lib_exc.NotFound:
        return True
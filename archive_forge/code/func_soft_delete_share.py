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
@not_found_wrapper
@forbidden_wrapper
def soft_delete_share(self, shares, microversion=None):
    """Soft Delete share[s] by Names or IDs.

        :param shares: either str or list of str that can be either Name
            or ID of a share(s) that should be soft deleted.
        """
    if not isinstance(shares, list):
        shares = [shares]
    cmd = 'soft-delete '
    for share in shares:
        cmd += '%s ' % share
    return self.manila(cmd, microversion=microversion)
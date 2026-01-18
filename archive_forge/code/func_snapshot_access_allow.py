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
def snapshot_access_allow(self, snapshot_id, access_type, access_to, microversion=None):
    raw_access = self.manila('snapshot-access-allow %(id)s %(type)s %(access_to)s' % {'id': snapshot_id, 'type': access_type, 'access_to': access_to}, microversion=microversion)
    return output_parser.details(raw_access)
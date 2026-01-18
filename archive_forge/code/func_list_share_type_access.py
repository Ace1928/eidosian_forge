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
def list_share_type_access(self, share_type_id, microversion=None):
    projects_raw = self.manila('type-access-list %s' % share_type_id, microversion=microversion)
    projects = output_parser.listing(projects_raw)
    project_ids = [pr['Project_ID'] for pr in projects]
    return project_ids
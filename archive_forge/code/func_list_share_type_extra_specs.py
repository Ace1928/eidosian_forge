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
def list_share_type_extra_specs(self, share_type_name_or_id, microversion=None):
    """List extra specs for specific share type by its Name or ID."""
    all_share_types = self.list_all_share_type_extra_specs(microversion=microversion)
    for share_type in all_share_types:
        if share_type_name_or_id in (share_type['ID'], share_type['Name']):
            return share_type['all_extra_specs']
    raise exceptions.ShareTypeNotFound(share_type=share_type_name_or_id)
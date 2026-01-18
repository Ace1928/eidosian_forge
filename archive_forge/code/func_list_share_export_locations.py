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
def list_share_export_locations(self, share, columns=None, microversion=None):
    """List share export locations.

        :param share: str -- Name or ID of a share.
        :param columns: str -- comma separated string of columns.
            Example, "--columns uuid,path".
        :param microversion: API microversion to be used for request.
        """
    cmd = 'share-export-location-list %s' % share
    if columns is not None:
        cmd += ' --columns ' + columns
    export_locations_raw = self.manila(cmd, microversion=microversion)
    export_locations = utils.listing(export_locations_raw)
    return export_locations
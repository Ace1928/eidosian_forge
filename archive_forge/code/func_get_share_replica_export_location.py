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
def get_share_replica_export_location(self, share_replica, export_location_uuid, microversion=None):
    """Returns an export location by share replica and export location ID.

        :param share_replica: str -- ID of share replica.
        :param export_location_uuid: str -- UUID of an export location.
        :param microversion: API microversion to be used for request.
        """
    export_raw = self.manila('share-replica-export-location-show %(share_replica)s %(el_uuid)s' % {'share_replica': share_replica, 'el_uuid': export_location_uuid}, microversion=microversion)
    export = output_parser.details(export_raw)
    return export
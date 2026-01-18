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
def get_share_instance_export_location(self, share_instance, export_location_uuid, microversion=None):
    """Returns an export location by share instance and its UUID.

        :param share_instance: str -- Name or ID of a share instance.
        :param export_location_uuid: str -- UUID of an export location.
        :param microversion: API microversion to be used for request.
        """
    share_raw = self.manila('share-instance-export-location-show %(share_instance)s %(el_uuid)s' % {'share_instance': share_instance, 'el_uuid': export_location_uuid}, microversion=microversion)
    share = output_parser.details(share_raw)
    return share
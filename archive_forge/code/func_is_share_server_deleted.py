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
def is_share_server_deleted(self, share_server_id, microversion=None):
    """Says whether share server is deleted or not.

        :param share_server: text -- ID of the share server
        """
    servers = self.list_share_servers(microversion=microversion)
    for list_element in servers:
        if share_server_id == list_element['Id']:
            return False
    return True
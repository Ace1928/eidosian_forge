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
def list_messages(self, columns=None, microversion=None):
    """List messages.

        :param columns: str -- comma separated string of columns.
            Example, "--columns id,resource_id".
        :param microversion: API microversion to be used for request.
        """
    cmd = 'message-list'
    if columns is not None:
        cmd += ' --columns ' + columns
    messages_raw = self.manila(cmd, microversion=microversion)
    messages = utils.listing(messages_raw)
    return messages
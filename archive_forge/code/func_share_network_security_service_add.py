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
def share_network_security_service_add(self, share_network_id, security_service_id, microversion=None):
    cmd = 'share-network-security-service-add %(network_id)s %(service_id)s' % {'network_id': share_network_id, 'service_id': security_service_id}
    self.manila(cmd, microversion=microversion)
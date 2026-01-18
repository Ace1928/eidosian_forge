import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from tempest.lib.cli import output_parser
from testtools import matchers
import manilaclient
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
def shell_discover_client(self, current_client, os_api_version, os_endpoint_type, os_service_type, client_args):
    return (current_client, manilaclient.API_MAX_VERSION)
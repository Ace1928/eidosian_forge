import argparse
import io
import json
import re
import sys
from unittest import mock
import ddt
import fixtures
import keystoneauth1.exceptions as ks_exc
from keystoneauth1.exceptions import DiscoveryFailure
from keystoneauth1.identity.generic.password import Password as ks_password
from keystoneauth1 import session
import requests_mock
from testtools import matchers
import cinderclient
from cinderclient import api_versions
from cinderclient.contrib import noauth
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit import fake_actions_module
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_help_arg_no_subcommand(self):
    required = ['.*?^usage: ', '.*?^\\s+create\\s+Creates a volume.', '.*?^\\s+summary\\s+Get volumes summary.', '.*?^Run "cinder help SUBCOMMAND" for help on a subcommand.']
    help_text = self.shell('--os-volume-api-version 3.40')
    for r in required:
        self.assertThat(help_text, matchers.MatchesRegex(r, re.DOTALL | re.MULTILINE))
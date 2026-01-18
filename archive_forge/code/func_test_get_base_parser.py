import argparse
from collections import OrderedDict
import hashlib
import io
import logging
import os
import sys
import traceback
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import fixture as ks_fixture
from requests_mock.contrib import fixture as rm_fixture
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell as openstack_shell
from glanceclient.tests.unit.v2.fixtures import image_show_fixture
from glanceclient.tests.unit.v2.fixtures import image_versions_fixture
from glanceclient.tests import utils as testutils
from glanceclient.v2 import schemas as schemas
import json
def test_get_base_parser(self):
    test_shell = openstack_shell.OpenStackImagesShell()
    actual_parser = test_shell.get_base_parser(sys.argv)
    description = 'Command-line interface to the OpenStack Images API.'
    expected = argparse.ArgumentParser(prog='glance', usage=None, description=description, conflict_handler='error', add_help=False, formatter_class=openstack_shell.HelpFormatter)
    self.assertEqual(str(expected), str(actual_parser))
import contextlib
import copy
import json as jsonutils
import os
from unittest import mock
from cliff import columns as cliff_columns
import fixtures
from keystoneauth1 import loading
from openstack.config import cloud_region
from openstack.config import defaults
from oslo_utils import importutils
from requests_mock.contrib import fixture
import testtools
from osc_lib import clientmanager
from osc_lib import shell
from osc_lib.tests import fakes
def opt2attr(opt):
    if opt.startswith('--os-'):
        attr = opt[5:]
    elif opt.startswith('--'):
        attr = opt[2:]
    else:
        attr = opt
    return attr.lower().replace('-', '_')
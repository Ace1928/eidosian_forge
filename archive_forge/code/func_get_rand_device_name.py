import logging as std_logging
import os
import os.path
import random
from unittest import mock
import fixtures
from oslo_config import cfg
from oslo_db import options as db_options
from oslo_utils import strutils
import pbr.version
import testtools
from neutron_lib._i18n import _
from neutron_lib import constants
from neutron_lib import exceptions
from neutron_lib import fixture
from neutron_lib.tests import _post_mortem_debug as post_mortem_debug
def get_rand_device_name(prefix='test'):
    return get_rand_name(max_length=constants.DEVICE_NAME_MAX_LEN, prefix=prefix)
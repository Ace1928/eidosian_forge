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
def get_rand_name(max_length=None, prefix='test'):
    """Return a random string.

    The string will start with 'prefix' and will be exactly 'max_length'.
    If 'max_length' is None, then exactly 8 random characters, each
    hexadecimal, will be added. In case len(prefix) <= len(max_length),
    ValueError will be raised to indicate the problem.
    """
    if max_length:
        length = max_length - len(prefix)
        if length <= 0:
            raise ValueError("'max_length' must be bigger than 'len(prefix)'.")
        suffix = ''.join((str(random.randint(0, 9)) for i in range(length)))
    else:
        suffix = hex(random.randint(268435456, 2147483647))[2:]
    return prefix + suffix
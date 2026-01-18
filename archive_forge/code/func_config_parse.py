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
@staticmethod
def config_parse(conf=None, args=None):
    """Create the default configurations."""
    if args is None:
        args = []
    args += ['--config-file', etcdir('neutron_lib.conf')]
    if conf is None:
        version_info = pbr.version.VersionInfo('neutron-lib')
        cfg.CONF(args=args, project='neutron_lib', version='%%(prog)s %s' % version_info.release_string())
    else:
        conf(args)
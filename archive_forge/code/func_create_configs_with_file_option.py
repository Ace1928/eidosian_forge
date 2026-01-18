import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
def create_configs_with_file_option(test):
    """Create configuration files with a ``file`` option set in each.

    This builds on ``create_configs`` and add one ``file`` option in each
    configuration with a value which allows identifying the configuration file.
    """
    create_configs(test)
    test.breezy_config.set_user_option('file', 'breezy')
    test.locations_config.set_user_option('file', 'locations')
    test.branch_config.set_user_option('file', 'branch')
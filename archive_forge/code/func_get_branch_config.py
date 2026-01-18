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
def get_branch_config(self, global_config=None, location=None, location_config=None, branch_data_config=None):
    my_branch = FakeBranch(location)
    if global_config is not None:
        config.GlobalConfig.from_string(global_config, save=True)
    if location_config is not None:
        config.LocationConfig.from_string(location_config, my_branch.base, save=True)
    my_config = config.BranchConfig(my_branch)
    if branch_data_config is not None:
        my_config.branch.control_files.files['branch.conf'] = branch_data_config
    return my_config
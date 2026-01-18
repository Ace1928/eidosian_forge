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
def register_bool_option(self, name, default=None, default_from_env=None):
    b = config.Option(name, help='A boolean.', default=default, default_from_env=default_from_env, from_unicode=config.bool_from_store)
    self.registry.register(b)
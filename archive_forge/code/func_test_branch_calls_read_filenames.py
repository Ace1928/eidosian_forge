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
def test_branch_calls_read_filenames(self):
    oldparserclass = config.ConfigObj
    config.ConfigObj = InstrumentedConfigObj
    try:
        my_config = config.LocationConfig('http://www.example.com')
        parser = my_config._get_parser()
    finally:
        config.ConfigObj = oldparserclass
    self.assertIsInstance(parser, InstrumentedConfigObj)
    self.assertEqual(parser._calls, [('__init__', bedding.locations_config_path(), 'utf-8')])
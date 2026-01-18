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
def assertConvertInvalid(self, opt, invalid_value):
    opt.invalid = None
    self.assertEqual(None, opt.convert_from_unicode(None, invalid_value))
    opt.invalid = 'warning'
    self.assertCallsWarning(opt, invalid_value)
    opt.invalid = 'error'
    self.assertCallsError(opt, invalid_value)
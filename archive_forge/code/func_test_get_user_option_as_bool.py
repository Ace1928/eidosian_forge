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
def test_get_user_option_as_bool(self):
    conf, parser = self.make_config_parser('\na_true_bool = true\na_false_bool = 0\nan_invalid_bool = maybe\na_list = hmm, who knows ? # This is interpreted as a list !\n')
    get_bool = conf.get_user_option_as_bool
    self.assertEqual(True, get_bool('a_true_bool'))
    self.assertEqual(False, get_bool('a_false_bool'))
    warnings = []

    def warning(*args):
        warnings.append(args[0] % args[1:])
    self.overrideAttr(trace, 'warning', warning)
    msg = 'Value "%s" is not a boolean for "%s"'
    self.assertIs(None, get_bool('an_invalid_bool'))
    self.assertEqual(msg % ('maybe', 'an_invalid_bool'), warnings[0])
    warnings = []
    self.assertIs(None, get_bool('not_defined_in_this_config'))
    self.assertEqual([], warnings)
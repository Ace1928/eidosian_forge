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
def test_hash_sign_in_value(self):
    """
        Before 4.5.0, ConfigObj did not quote # signs in values, so they'd be
        treated as comments when read in again. (#86838)
        """
    co = config.ConfigObj()
    co['test'] = 'foo#bar'
    outfile = BytesIO()
    co.write(outfile=outfile)
    lines = outfile.getvalue().splitlines()
    self.assertEqual(lines, [b'test = "foo#bar"'])
    co2 = config.ConfigObj(lines)
    self.assertEqual(co2['test'], 'foo#bar')
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
def test_get_change_editor(self):
    my_config = self._get_sample_config()
    change_editor = my_config.get_change_editor('old', 'new')
    self.assertIs(diff.DiffFromTool, change_editor.__class__)
    self.assertEqual('vimdiff -of {new_path} {old_path}', ' '.join(change_editor.command_template))
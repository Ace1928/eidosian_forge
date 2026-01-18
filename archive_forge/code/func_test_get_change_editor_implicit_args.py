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
def test_get_change_editor_implicit_args(self):
    my_config = InstrumentedConfig()
    my_config._change_editor = 'vimdiff -o'
    change_editor = my_config.get_change_editor('old_tree', 'new_tree')
    self.assertEqual(['_get_change_editor'], my_config._calls)
    self.assertIs(diff.DiffFromTool, change_editor.__class__)
    self.assertEqual(['vimdiff', '-o', '{old_path}', '{new_path}'], change_editor.command_template)
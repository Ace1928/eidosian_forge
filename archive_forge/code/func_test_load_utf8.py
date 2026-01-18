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
def test_load_utf8(self):
    """Ensure we can load an utf8-encoded file."""
    unicode_user = 'b€ar'
    unicode_content = 'user={}'.format(unicode_user)
    utf8_content = unicode_content.encode('utf8')
    with open('foo.conf', 'wb') as f:
        f.write(utf8_content)
    conf = config.IniBasedConfig(file_name='foo.conf')
    self.assertEqual(unicode_user, conf.get_user_option('user'))
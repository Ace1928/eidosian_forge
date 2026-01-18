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
def test_for_matching_host(self):
    conf = config.AuthenticationConfig(_file=BytesIO(b'# Identity on foo.net\n[sourceforge]\nscheme=bzr\nhost=bzr.sf.net\nuser=joe\npassword=joepass\n[sourceforge domain]\nscheme=bzr\nhost=.bzr.sf.net\nuser=georges\npassword=bendover\n'))
    self._got_user_passwd('georges', 'bendover', conf, 'bzr', 'foo.bzr.sf.net')
    self._got_user_passwd(None, None, conf, 'bzr', 'bbzr.sf.net')
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
def test_credentials_for_path(self):
    conf = config.AuthenticationConfig(_file=BytesIO(b'\n[http dir1]\nscheme=http\nhost=bar.org\npath=/dir1\nuser=jim\npassword=jimpass\n[http dir2]\nscheme=http\nhost=bar.org\npath=/dir2\nuser=georges\npassword=bendover\n'))
    self._got_user_passwd(None, None, conf, 'http', host='bar.org', path='/dir3')
    self._got_user_passwd('georges', 'bendover', conf, 'http', host='bar.org', path='/dir2')
    self._got_user_passwd('jim', 'jimpass', conf, 'http', host='bar.org', path='/dir1/subdir')
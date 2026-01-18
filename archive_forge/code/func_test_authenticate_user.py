import io
import re
import sys
import fixtures
import testtools
from blazarclient import shell
from blazarclient import tests
@testtools.skip('lol')
def test_authenticate_user(self):
    obj = shell.BlazarShell()
    obj.initialize_app('list-leases')
    obj.options.os_token = 'aaaa-bbbb-cccc'
    obj.options.os_cacert = 'cert'
    obj.authenticate_user()
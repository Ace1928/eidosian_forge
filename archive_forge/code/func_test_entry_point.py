import stevedore
from testtools import matchers
from keystonemiddleware.auth_token import _opts as new_opts
from keystonemiddleware import opts as old_opts
from keystonemiddleware.tests.unit import utils
def test_entry_point(self):
    em = stevedore.ExtensionManager('oslo.config.opts', invoke_on_load=True)
    for extension in em:
        if extension.name == 'keystonemiddleware.auth_token':
            break
    else:
        self.fail('keystonemiddleware.auth_token not found')
    self._test_list_auth_token_opts(extension.obj)
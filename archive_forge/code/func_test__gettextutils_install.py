import builtins
import gettext
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _factory
from oslo_i18n import _gettextutils
from oslo_i18n import _lazy
from oslo_i18n import _message
def test__gettextutils_install(self):
    _gettextutils.install('blaa')
    _lazy.enable_lazy(False)
    self.assertTrue(isinstance(self.t.primary('A String'), str))
    _gettextutils.install('blaa')
    _lazy.enable_lazy(True)
    self.assertTrue(isinstance(self.t.primary('A Message'), _message.Message))
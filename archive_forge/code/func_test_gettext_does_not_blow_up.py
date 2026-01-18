import builtins
import gettext
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _factory
from oslo_i18n import _gettextutils
from oslo_i18n import _lazy
from oslo_i18n import _message
def test_gettext_does_not_blow_up(self):
    LOG.info(self.t.primary('test'))
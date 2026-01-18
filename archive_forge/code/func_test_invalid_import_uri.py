import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
def test_invalid_import_uri(self):
    self.assertFalse(utils.validate_import_uri(''))
    self.assertFalse(utils.validate_import_uri('fake_uri'))
    self.config(disallowed_schemes=['ftp'], group='import_filtering_opts')
    self.assertFalse(utils.validate_import_uri('ftp://example.com'))
    self.config(disallowed_hosts=['foo.com'], group='import_filtering_opts')
    self.assertFalse(utils.validate_import_uri('ftp://foo.com'))
    self.config(disallowed_ports=['8484'], group='import_filtering_opts')
    self.assertFalse(utils.validate_import_uri('http://localhost:8484'))
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
def test_ignored_filtering_options(self):
    LOG = logging.getLogger('glance.common.utils')
    with mock.patch.object(LOG, 'debug') as mock_run:
        self.config(allowed_schemes=['https', 'ftp'], group='import_filtering_opts')
        self.config(disallowed_schemes=['ftp'], group='import_filtering_opts')
        self.assertTrue(utils.validate_import_uri('ftp://foo.com'))
        mock_run.assert_called_once()
    with mock.patch.object(LOG, 'debug') as mock_run:
        self.config(allowed_schemes=[], group='import_filtering_opts')
        self.config(disallowed_schemes=[], group='import_filtering_opts')
        self.config(allowed_hosts=['example.com', 'foo.com'], group='import_filtering_opts')
        self.config(disallowed_hosts=['foo.com'], group='import_filtering_opts')
        self.assertTrue(utils.validate_import_uri('ftp://foo.com'))
        mock_run.assert_called_once()
    with mock.patch.object(LOG, 'debug') as mock_run:
        self.config(allowed_hosts=[], group='import_filtering_opts')
        self.config(disallowed_hosts=[], group='import_filtering_opts')
        self.config(allowed_ports=[8080, 8484], group='import_filtering_opts')
        self.config(disallowed_ports=[8484], group='import_filtering_opts')
        self.assertTrue(utils.validate_import_uri('ftp://foo.com:8484'))
        mock_run.assert_called_once()
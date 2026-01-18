from unittest import mock
import fixtures
from keystonemiddleware.tests.unit.audit import base
def test_api_request_no_messaging(self):
    self.cfg.config(use_oslo_messaging=False, group='audit_middleware_notifications')
    app = self.create_simple_app()
    with mock.patch('keystonemiddleware.audit._LOG.info') as log:
        app.get('/foo/bar', extra_environ=self.get_environ_header())
        call_args = log.call_args_list[0][0]
        self.assertEqual('audit.http.request', call_args[1]['event_type'])
        call_args = log.call_args_list[1][0]
        self.assertEqual('audit.http.response', call_args[1]['event_type'])
from unittest import mock
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_conf_middleware_messaging_and_oslo_msg_as_log(self):
    self.cfg.config(driver=['log'], group='oslo_messaging_notifications')
    self.cfg.config(driver='messaging', group='audit_middleware_notifications')
    app = self.create_simple_app()
    with mock.patch('oslo_messaging.notify.messaging.MessagingDriver.notify', side_effect=Exception('error')) as driver:
        app.get('/foo/bar', extra_environ=self.get_environ_header())
        self.assertTrue(driver.called)
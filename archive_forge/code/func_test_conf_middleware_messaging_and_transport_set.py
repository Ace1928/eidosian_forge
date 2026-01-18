from unittest import mock
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
@mock.patch('oslo_messaging.get_notification_transport')
def test_conf_middleware_messaging_and_transport_set(self, m):
    transport_url = 'rabbit://me:passwd@host:5672/virtual_host'
    self.cfg.config(driver='messaging', transport_url=transport_url, group='audit_middleware_notifications')
    self.create_simple_middleware()
    self.assertTrue(m.called)
    self.assertEqual(transport_url, m.call_args_list[0][1]['url'])
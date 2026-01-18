from unittest import mock
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_do_not_use_oslo_messaging(self):
    self.cfg.config(use_oslo_messaging=False, group='audit_middleware_notifications')
    audit_middleware = self.create_simple_middleware()
    self.assertIsInstance(audit_middleware._notifier, audit._notifier._LogNotifier)
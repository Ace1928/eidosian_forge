from unittest import mock
import uuid
import fixtures
import webob
from keystonemiddleware.tests.unit.audit import base
def test_cadf_event_context_scoped(self):
    self.create_simple_app().get('/foo/bar', extra_environ=self.get_environ_header())
    self.assertEqual(2, self.notifier.notify.call_count)
    first, second = [a[0] for a in self.notifier.notify.call_args_list]
    self.assertIsInstance(first[0], dict)
    self.assertIs(first[0], second[0])
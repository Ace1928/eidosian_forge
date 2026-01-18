import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_body_hook_chaining(self):
    test_hook1 = TestHook('foo')
    test_hook2 = TestHook('bar')
    client = self.compose_with_hooks([test_hook1, test_hook2])[0]
    self.assertEqual(None, test_hook1.calls[0].body)
    self.assertEqual(None, test_hook1.calls[0].orig_body)
    self.assertEqual('foo', test_hook2.calls[0].body)
    self.assertEqual(None, test_hook2.calls[0].orig_body)
    self.assertEqual('bar', client.body)
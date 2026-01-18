import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_no_supports_body(self):
    test_hook = TestHook('foo')
    old_warn = trace.warning
    warnings = []

    def warn(*args):
        warnings.append(args)
    trace.warning = warn
    try:
        client, directive = self.compose_with_hooks([test_hook], supports_body=False)
    finally:
        trace.warning = old_warn
    self.assertEqual(0, len(test_hook.calls))
    self.assertEqual(('Cannot run merge_request_body hooks because mail client %s does not support message bodies.', 'HookMailClient'), warnings[0])
def assert_in_call(self, url_part):
    """Assert a call contained a part in its URL."""
    assert self.client.callstack, 'Expected call but no calls were made'
    called = self.client.callstack[-1][1]
    assert url_part in called, 'Expected %s in call but found %s' % (url_part, called)
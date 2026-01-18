from unittest import mock
from keystoneauth1 import plugin
def assert_called_anytime(self, method, url, body=None):
    """Assert that an API method was called anytime in the test."""
    expected = (method, url)
    if not self.client.callstack:
        raise AssertionError('Expected %s %s but no calls were made.' % expected)
    found = False
    for entry in self.client.callstack:
        if expected == entry[0:2]:
            found = True
            break
    if not found:
        raise AssertionError('Expected %s; got %s' % (expected, self.client.callstack))
    if body is not None:
        if entry[2] != body:
            raise AssertionError('%s != %s' % (entry[2], body))
    self.client.callstack = []
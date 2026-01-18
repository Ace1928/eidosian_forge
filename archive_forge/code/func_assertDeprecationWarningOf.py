from twisted.trial import unittest
from twisted.web import html
def assertDeprecationWarningOf(method: str) -> None:
    """
            Check that a deprecation warning is present.
            """
    warningsShown = self.flushWarnings([self.test_deprecation])
    self.assertEqual(len(warningsShown), 1)
    self.assertIdentical(warningsShown[0]['category'], DeprecationWarning)
    self.assertEqual(warningsShown[0]['message'], 'twisted.web.html.%s was deprecated in Twisted 15.3.0; please use twisted.web.template instead' % (method,))
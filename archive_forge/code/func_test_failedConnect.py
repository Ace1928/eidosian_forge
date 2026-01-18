from twisted.internet import defer, error
from twisted.trial import unittest
from twisted.words.im import basesupport
def test_failedConnect(self):
    """
        Test that account.logOn works, and it calls the right callback when a
        connection is established.
        """
    account = self.makeAccount()
    ui = self.makeUI()
    d = account.logOn(ui)
    account.loginDeferred.errback(Exception())

    def err(reason):
        self.assertTrue(account.loginHasFailed, 'Login should have failed')
        self.assertFalse(account.loginCallbackCalled, "We shouldn't be logged in")
        self.assertTrue(not ui.clientRegistered, "Client shouldn't be registered in the UI")
    cb = lambda r: self.assertTrue(False, "Shouldn't get called back")
    d.addCallbacks(cb, err)
    return d
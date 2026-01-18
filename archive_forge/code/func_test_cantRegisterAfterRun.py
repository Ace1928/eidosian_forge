from __future__ import annotations
from unittest import skipIf
from twisted.internet.error import ReactorAlreadyRunning
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.trial.unittest import SkipTest, TestCase
def test_cantRegisterAfterRun(self) -> None:
    """
        It is not possible to register a C{Application} after the reactor has
        already started.
        """
    self.reactorFactory = lambda: gireactor.GIReactor(useGtk=False)
    reactor = self.buildReactor()
    app = Gio.Application(application_id='com.twistedmatrix.trial.gireactor', flags=Gio.ApplicationFlags.FLAGS_NONE)

    def tryRegister() -> None:
        exc = self.assertRaises(ReactorAlreadyRunning, reactor.registerGApplication, app)
        self.assertEqual(exc.args[0], "Can't register application after reactor was started.")
        reactor.stop()
    reactor.callLater(0, tryRegister)
    ReactorBuilder.runReactor(self, reactor)
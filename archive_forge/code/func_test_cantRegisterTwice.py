from __future__ import annotations
from unittest import skipIf
from twisted.internet.error import ReactorAlreadyRunning
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.trial.unittest import SkipTest, TestCase
def test_cantRegisterTwice(self) -> None:
    """
        It is not possible to register more than one C{Application}.
        """
    self.reactorFactory = lambda: gireactor.GIReactor(useGtk=False)
    reactor = self.buildReactor()
    app = Gio.Application(application_id='com.twistedmatrix.trial.gireactor', flags=Gio.ApplicationFlags.FLAGS_NONE)
    reactor.registerGApplication(app)
    app2 = Gio.Application(application_id='com.twistedmatrix.trial.gireactor2', flags=Gio.ApplicationFlags.FLAGS_NONE)
    exc = self.assertRaises(RuntimeError, reactor.registerGApplication, app2)
    self.assertEqual(exc.args[0], "Can't register more than one application instance.")
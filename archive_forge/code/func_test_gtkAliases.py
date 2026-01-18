from __future__ import annotations
from unittest import skipIf
from twisted.internet.error import ReactorAlreadyRunning
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.trial.unittest import SkipTest, TestCase
@skipIf(noGtkSkip, noGtkMessage)
def test_gtkAliases(self) -> None:
    """
        L{twisted.internet.gtk3reactor} is now just a set of compatibility
        aliases for L{twisted.internet.GIReactor}.
        """
    from twisted.internet.gtk3reactor import Gtk3Reactor, PortableGtk3Reactor, install
    self.assertIs(Gtk3Reactor, gireactor.GIReactor)
    self.assertIs(PortableGtk3Reactor, gireactor.PortableGIReactor)
    self.assertIs(install, gireactor.install)
    warnings = self.flushWarnings()
    self.assertEqual(len(warnings), 1)
    self.assertIn('twisted.internet.gtk3reactor was deprecated', warnings[0]['message'])
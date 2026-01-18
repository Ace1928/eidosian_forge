from __future__ import annotations
from twisted.cred import credentials, error
from twisted.cred.checkers import FilePasswordDB
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words import tap
def test_auth(self) -> None:
    """
        Tests that the --auth command generates a checker.
        """
    opt = tap.Options()
    opt.parseOptions(['--auth', 'file:' + self.file.name])
    self._loginTest(opt)
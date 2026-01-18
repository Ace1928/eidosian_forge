from twisted.trial import unittest
def test_manual(self) -> None:
    """
        Calling L{appdirs.getDataDirectory} with a C{moduleName} argument will
        make a data directory with that name instead.
        """
    res = _appdirs.getDataDirectory('foo.bar.baz')
    self.assertTrue(res.endswith('foo.bar.baz'))
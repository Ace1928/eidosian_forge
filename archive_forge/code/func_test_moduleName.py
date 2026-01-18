from twisted.trial import unittest
def test_moduleName(self) -> None:
    """
        Calling L{appdirs.getDataDirectory} will return a user data directory
        in the system convention, with the module of the caller as the
        subdirectory.
        """
    res = _appdirs.getDataDirectory()
    self.assertTrue(res.endswith('twisted.python.test.test_appdirs'))
from twisted.trial import unittest
from .._levels import InvalidLogLevelError, LogLevel
def test_levelWithName(self) -> None:
    """
        Look up log level by name.
        """
    for level in LogLevel.iterconstants():
        self.assertIs(LogLevel.levelWithName(level.name), level)
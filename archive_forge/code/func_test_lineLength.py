from io import StringIO
from twisted.python import text
from twisted.trial import unittest
def test_lineLength(self) -> None:
    """
        Check the length of the lines.
        """
    failures = []
    for line in self.output:
        if not len(line) <= self.lineWidth:
            failures.append(len(line))
    if failures:
        self.fail('%d of %d lines were too long.\n%d < %s' % (len(failures), len(self.output), self.lineWidth, failures))
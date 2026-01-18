import sys
import time
import warnings
from . import result
from .case import _SubTest
from .signals import registerResult
def printErrorList(self, flavour, errors):
    for test, err in errors:
        self.stream.writeln(self.separator1)
        self.stream.writeln('%s: %s' % (flavour, self.getDescription(test)))
        self.stream.writeln(self.separator2)
        self.stream.writeln('%s' % err)
        self.stream.flush()
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def openlog(self, prefix, options, facility):
    self.logOpened = (prefix, options, facility)
    self.events = []
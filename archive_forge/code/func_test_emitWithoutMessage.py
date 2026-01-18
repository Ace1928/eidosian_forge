from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_emitWithoutMessage(self):
    """
        L{SyslogObserver.emit} ignores events with an empty value for the
        C{'message'} key.
        """
    self.observer.emit({'message': (), 'isError': False, 'system': '-'})
    self.assertEqual(self.events, [])
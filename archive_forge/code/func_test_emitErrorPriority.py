from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_emitErrorPriority(self):
    """
        L{SyslogObserver.emit} uses C{LOG_ALERT} if the event represents an
        error.
        """
    self.observer.emit({'message': ('hello, world',), 'isError': True, 'system': '-', 'failure': Failure(Exception('foo'))})
    self.assertEqual(self.events, [(stdsyslog.LOG_ALERT, '[-] hello, world')])
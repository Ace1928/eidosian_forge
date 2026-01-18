from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_failedInit(self):
    """
        If C{inotify_init} returns a negative number, L{init} raises
        L{INotifyError}.
        """

    class libc:

        def inotify_init(self):
            return -1
    self.patch(inotify, 'libc', libc())
    self.assertRaises(INotifyError, init)
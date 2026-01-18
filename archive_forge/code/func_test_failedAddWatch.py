from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_failedAddWatch(self):
    """
        If C{inotify_add_watch} returns a negative number, L{add}
        raises L{INotifyError}.
        """

    class libc:

        def inotify_add_watch(self, fd, path, mask):
            return -1
    self.patch(inotify, 'libc', libc())
    self.assertRaises(INotifyError, add, 3, FilePath('/foo'), 0)
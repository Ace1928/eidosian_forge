import sys
import unittest
from unittest import mock
from bpython.curtsiesfrontend.coderunner import CodeRunner, FakeOutput
def test_exception(self):
    c = CodeRunner(request_refresh=lambda: self.orig_stdout.flush() or self.orig_stderr.flush())

    def ctrlc():
        raise KeyboardInterrupt()
    stdout = FakeOutput(c, lambda x: ctrlc(), None)
    stderr = FakeOutput(c, lambda *args, **kwargs: None, None)
    sys.stdout = stdout
    sys.stderr = stderr
    c.load_code('1 + 1')
    c.run_code()
import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
def test_function_signature(self):
    fake_signature = inspect.getfullargspec(FakePopen.__call__)
    real_signature = inspect.getfullargspec(subprocess.Popen)
    fake_args = fake_signature.args
    real_args = real_signature.args
    self.assertListEqual(fake_args, real_args, "Function signature of FakePopen doesn't match subprocess.Popen")
    fake_kwargs = set(fake_signature.kwonlyargs)
    real_kwargs = set(real_signature.kwonlyargs)
    if sys.version_info < (3, 11):
        fake_kwargs.remove('process_group')
    if sys.version_info < (3, 10):
        fake_kwargs.remove('pipesize')
    if sys.version_info < (3, 9):
        fake_kwargs.remove('group')
        fake_kwargs.remove('extra_groups')
        fake_kwargs.remove('user')
        fake_kwargs.remove('umask')
    if sys.version_info < (3, 7):
        fake_kwargs.remove('text')
    self.assertSetEqual(fake_kwargs, real_kwargs, "Function signature of FakePopen doesn't match subprocess.Popen")
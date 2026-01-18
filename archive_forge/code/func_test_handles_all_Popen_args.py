import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
def test_handles_all_Popen_args(self):
    all_args = dict(args='args', bufsize='bufsize', executable='executable', stdin='stdin', stdout='stdout', stderr='stderr', preexec_fn='preexec_fn', close_fds='close_fds', shell='shell', cwd='cwd', env='env', universal_newlines='universal_newlines', startupinfo='startupinfo', creationflags='creationflags', restore_signals='restore_signals', start_new_session='start_new_session', pass_fds='pass_fds', encoding='encoding', errors='errors')
    if sys.version_info >= (3, 7):
        all_args['text'] = 'text'
    if sys.version_info >= (3, 9):
        all_args['group'] = 'group'
        all_args['extra_groups'] = 'extra_groups'
        all_args['user'] = 'user'
        all_args['umask'] = 'umask'
    if sys.version_info >= (3, 10):
        all_args['pipesize'] = 'pipesize'
    if sys.version_info >= (3, 11):
        all_args['process_group'] = 'process_group'

    def get_info(proc_args):
        self.assertEqual(all_args, proc_args)
        return {}
    fixture = self.useFixture(FakePopen(get_info))
    fixture(**all_args)
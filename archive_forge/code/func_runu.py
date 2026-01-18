import sys
import types
from .exceptions import EOF, TIMEOUT
from .pty_spawn import spawn
def runu(command, timeout=30, withexitstatus=False, events=None, extra_args=None, logfile=None, cwd=None, env=None, **kwargs):
    """Deprecated: pass encoding to run() instead.
    """
    kwargs.setdefault('encoding', 'utf-8')
    return run(command, timeout=timeout, withexitstatus=withexitstatus, events=events, extra_args=extra_args, logfile=logfile, cwd=cwd, env=env, **kwargs)
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from os.path import abspath
from os.path import join as pjoin
from subprocess import STDOUT, check_call, check_output
from ._in_process import _in_proc_script_path
@contextmanager
def subprocess_runner(self, runner):
    """A context manager for temporarily overriding the default
        :ref:`subprocess runner <Subprocess Runners>`.

        .. code-block:: python

            hook_caller = BuildBackendHookCaller(...)
            with hook_caller.subprocess_runner(quiet_subprocess_runner):
                ...
        """
    prev = self._subprocess_runner
    self._subprocess_runner = runner
    try:
        yield
    finally:
        self._subprocess_runner = prev
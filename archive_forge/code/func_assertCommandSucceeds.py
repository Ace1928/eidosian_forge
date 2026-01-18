import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def assertCommandSucceeds(self, command, regexes=[''], env=None, close_fds=True):
    """Asserts that a shell command succeeds (i.e. exits with code 0).

    Args:
      command: List or string representing the command to run.
      regexes: List of regular expression strings.
      env: Dictionary of environment variable settings.
      close_fds: Whether or not to close all open fd's in the child after
        forking.
    """
    ret_code, err = GetCommandStderr(command, env, close_fds)
    command_string = GetCommandString(command)
    self.assert_(ret_code == 0, 'Running command\n%s failed with error code %s and message\n%s' % (_QuoteLongString(command_string), ret_code, _QuoteLongString(err)))
    self.assertRegexMatch(err, regexes, message='Running command\n%s failed with error code %s and message\n%s which matches no regex in %s' % (_QuoteLongString(command_string), ret_code, _QuoteLongString(err), regexes))
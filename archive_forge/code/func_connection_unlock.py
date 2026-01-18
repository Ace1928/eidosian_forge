from __future__ import (annotations, absolute_import, division, print_function)
import collections.abc as c
import fcntl
import io
import os
import shlex
import typing as t
from abc import abstractmethod
from functools import wraps
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.playbook.play_context import PlayContext
from ansible.plugins import AnsiblePlugin
from ansible.plugins.become import BecomeBase
from ansible.plugins.shell import ShellBase
from ansible.utils.display import Display
from ansible.plugins.loader import connection_loader, get_shell_plugin
from ansible.utils.path import unfrackpath
def connection_unlock(self) -> None:
    f = self._play_context.connection_lockfd
    fcntl.lockf(f, fcntl.LOCK_UN)
    display.vvvv('CONNECTION: pid %d released lock on %d' % (os.getpid(), f), host=self._play_context.remote_addr)
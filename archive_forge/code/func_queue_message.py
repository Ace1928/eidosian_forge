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
def queue_message(self, level: str, message: str) -> None:
    """
        Adds a message to the queue of messages waiting to be pushed back to the controller process.

        :arg level: A string which can either be the name of a method in display, or 'log'. When
            the messages are returned to task_executor, a value of log will correspond to
            ``display.display(message, log_only=True)``, while another value will call ``display.[level](message)``
        """
    self._messages.append((level, message))
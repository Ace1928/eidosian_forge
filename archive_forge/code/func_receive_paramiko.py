from __future__ import absolute_import, division, print_function
import getpass
import json
import logging
import os
import re
import signal
import socket
import time
import traceback
from functools import wraps
from io import BytesIO
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import cPickle
from ansible.playbook.play_context import PlayContext
from ansible.plugins.loader import cache_loader, cliconf_loader, connection_loader, terminal_loader
from ansible_collections.ansible.netcommon.plugins.connection.libssh import HAS_PYLIBSSH
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.connection_base import (
def receive_paramiko(self, command=None, prompts=None, answer=None, newline=True, prompt_retry_check=False, check_all=False, strip_prompt=True):
    recv = BytesIO()
    cache_socket_timeout = self.get_option('persistent_command_timeout')
    self._ssh_shell.settimeout(cache_socket_timeout)
    command_prompt_matched = False
    handled = False
    errored_response = None
    while True:
        if command_prompt_matched:
            try:
                signal.signal(signal.SIGALRM, self._handle_buffer_read_timeout)
                signal.setitimer(signal.ITIMER_REAL, self._buffer_read_timeout)
                data = self._ssh_shell.recv(256)
                signal.alarm(0)
                self._log_messages('response-%s: %s' % (self._window_count + 1, data))
                command_prompt_matched = False
                signal.signal(signal.SIGALRM, self._handle_command_timeout)
                signal.alarm(self._command_timeout)
            except AnsibleCmdRespRecv:
                return self._command_response
        else:
            data = self._ssh_shell.recv(256)
            self._log_messages('response-%s: %s' % (self._window_count + 1, data))
        if not data:
            break
        recv.write(data)
        offset = recv.tell() - 256 if recv.tell() > 256 else 0
        recv.seek(offset)
        window = self._strip(recv.read())
        self._last_recv_window = window
        self._window_count += 1
        if prompts and (not handled):
            handled = self._handle_prompt(window, prompts, answer, newline, False, check_all)
            self._matched_prompt_window = self._window_count
        elif prompts and handled and prompt_retry_check and (self._matched_prompt_window + 1 == self._window_count):
            if self._handle_prompt(window, prompts, answer, newline, prompt_retry_check, check_all):
                raise AnsibleConnectionFailure("For matched prompt '%s', answer is not valid" % self._matched_cmd_prompt)
        if self._find_error(window):
            errored_response = window
        if self._find_prompt(window):
            if errored_response:
                raise AnsibleConnectionFailure(errored_response)
            self._last_response = recv.getvalue()
            resp = self._strip(self._last_response)
            self._command_response = self._sanitize(resp, command, strip_prompt)
            if self._buffer_read_timeout == 0.0:
                return self._command_response
            else:
                command_prompt_matched = True
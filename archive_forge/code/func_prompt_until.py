from __future__ import annotations
import collections.abc as c
import codecs
import ctypes.util
import fcntl
import getpass
import io
import logging
import os
import random
import subprocess
import sys
import termios
import textwrap
import threading
import time
import tty
import typing as t
from functools import wraps
from struct import unpack, pack
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAssertionError, AnsiblePromptInterrupt, AnsiblePromptNoninteractive
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.six import text_type
from ansible.utils.color import stringc
from ansible.utils.multiprocessing import context as multiprocessing_context
from ansible.utils.singleton import Singleton
from ansible.utils.unsafe_proxy import wrap_var
def prompt_until(self, msg: str, private: bool=False, seconds: int | None=None, interrupt_input: c.Container[bytes] | None=None, complete_input: c.Container[bytes] | None=None) -> bytes:
    if self._final_q:
        from ansible.executor.process.worker import current_worker
        self._final_q.send_prompt(worker_id=current_worker.worker_id, prompt=msg, private=private, seconds=seconds, interrupt_input=interrupt_input, complete_input=complete_input)
        return current_worker.worker_queue.get()
    if HAS_CURSES and (not self.setup_curses):
        setupterm()
        self.setup_curses = True
    if self._stdin_fd is None or not os.isatty(self._stdin_fd) or os.getpgrp() != os.tcgetpgrp(self._stdin_fd):
        raise AnsiblePromptNoninteractive('stdin is not interactive')
    self.display(msg)
    result = b''
    with self._lock:
        original_stdin_settings = termios.tcgetattr(self._stdin_fd)
        try:
            setup_prompt(self._stdin_fd, self._stdout_fd, seconds, not private)
            termios.tcflush(self._stdin, termios.TCIFLUSH)
            return self._read_non_blocking_stdin(echo=not private, seconds=seconds, interrupt_input=interrupt_input, complete_input=complete_input)
        finally:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, original_stdin_settings)
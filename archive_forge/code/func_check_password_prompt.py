from __future__ import (absolute_import, division, print_function)
import shlex
from abc import abstractmethod
from random import choice
from string import ascii_lowercase
from gettext import dgettext
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes
from ansible.plugins import AnsiblePlugin
def check_password_prompt(self, b_output):
    """ checks if the expected password prompt exists in b_output """
    if self.prompt:
        b_prompt = to_bytes(self.prompt).strip()
        return any((l.strip().startswith(b_prompt) for l in b_output.splitlines()))
    return False
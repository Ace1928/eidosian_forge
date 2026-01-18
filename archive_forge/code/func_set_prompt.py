from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import atexit
import cmd
import getpass
import readline
import os
import sys
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.executor.task_queue_manager import TaskQueueManager
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.parsing.splitter import parse_kv
from ansible.playbook.play import Play
from ansible.plugins.list import list_plugins
from ansible.plugins.loader import module_loader, fragment_loader
from ansible.utils import plugin_docs
from ansible.utils.color import stringc
from ansible.utils.display import Display
def set_prompt(self):
    login_user = self.remote_user or getpass.getuser()
    self.selected = self.inventory.list_hosts(self.cwd)
    prompt = '%s@%s (%d)[f:%s]' % (login_user, self.cwd, len(self.selected), self.forks)
    if self.become and self.become_user in [None, 'root']:
        prompt += '# '
        color = C.COLOR_ERROR
    else:
        prompt += '$ '
        color = self.NORMAL_PROMPT
    self.prompt = stringc(prompt, color, wrap_nonvisible_chars=True)
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
def help_become_method(self):
    display.display('Set the privilege escalation plugin to use when become is enabled')
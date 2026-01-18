from __future__ import (absolute_import, division, print_function)
import cmd
import functools
import os
import pprint
import queue
import sys
import threading
import time
import typing as t
from collections import deque
from multiprocessing import Lock
from jinja2.exceptions import UndefinedError
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleFileNotFound, AnsibleUndefinedVariable, AnsibleParserError
from ansible.executor import action_write_locks
from ansible.executor.play_iterator import IteratingStates, PlayIterator
from ansible.executor.process.worker import WorkerProcess
from ansible.executor.task_result import TaskResult
from ansible.executor.task_queue_manager import CallbackSend, DisplaySend, PromptSend
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.playbook.conditional import Conditional
from ansible.playbook.handler import Handler
from ansible.playbook.helpers import load_list_of_blocks
from ansible.playbook.task import Task
from ansible.playbook.task_include import TaskInclude
from ansible.plugins import loader as plugin_loader
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.fqcn import add_internal_fqcns
from ansible.utils.unsafe_proxy import wrap_var
from ansible.utils.sentinel import Sentinel
from ansible.utils.vars import combine_vars, isidentifier
from ansible.vars.clean import strip_internal_keys, module_response_deepcopy
def post_process_whens(result, task, templar, task_vars):
    cond = None
    if task.changed_when:
        with templar.set_temporary_context(available_variables=task_vars):
            cond = Conditional(loader=templar._loader)
            cond.when = task.changed_when
            result['changed'] = cond.evaluate_conditional(templar, templar.available_variables)
    if task.failed_when:
        with templar.set_temporary_context(available_variables=task_vars):
            if cond is None:
                cond = Conditional(loader=templar._loader)
            cond.when = task.failed_when
            failed_when_result = cond.evaluate_conditional(templar, templar.available_variables)
            result['failed_when_result'] = result['failed'] = failed_when_result
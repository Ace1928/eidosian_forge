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
def normalize_task_result(self, task_result):
    """Normalize a TaskResult to reference actual Host and Task objects
        when only given the ``Host.name``, or the ``Task._uuid``

        Only the ``Host.name`` and ``Task._uuid`` are commonly sent back from
        the ``TaskExecutor`` or ``WorkerProcess`` due to performance concerns

        Mutates the original object
        """
    if isinstance(task_result._host, string_types):
        task_result._host = self._inventory.get_host(to_text(task_result._host))
    if isinstance(task_result._task, string_types):
        queue_cache_entry = (task_result._host.name, task_result._task)
        try:
            found_task = self._queued_task_cache[queue_cache_entry]['task']
        except KeyError:
            if task_result._task_fields.get('action') != 'async_status':
                raise
            original_task = Task()
        else:
            original_task = found_task.copy(exclude_parent=True, exclude_tasks=True)
            original_task._parent = found_task._parent
        original_task.from_attrs(task_result._task_fields)
        task_result._task = original_task
    return task_result
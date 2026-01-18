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
def search_handlers_by_notification(self, notification: str, iterator: PlayIterator) -> t.Generator[Handler, None, None]:
    templar = Templar(None)
    handlers = [h for b in reversed(iterator._play.handlers) for h in b.block]
    for handler in handlers:
        if not handler.name:
            continue
        if not handler.cached_name:
            if templar.is_template(handler.name):
                templar.available_variables = self._variable_manager.get_vars(play=iterator._play, task=handler, _hosts=self._hosts_cache, _hosts_all=self._hosts_cache_all)
                try:
                    handler.name = templar.template(handler.name)
                except (UndefinedError, AnsibleUndefinedVariable) as e:
                    if not handler.listen:
                        display.warning("Handler '%s' is unusable because it has no listen topics and the name could not be templated (host-specific variables are not supported in handler names). The error: %s" % (handler.name, to_text(e)))
                    continue
            handler.cached_name = True
        if notification in {handler.name, handler.get_name(include_role_fqcn=False), handler.get_name(include_role_fqcn=True)}:
            yield handler
            break
    templar.available_variables = {}
    seen = []
    for handler in handlers:
        if (listeners := handler.listen):
            if notification in handler.get_validated_value('listen', handler.fattributes.get('listen'), listeners, templar):
                if handler.name and handler.name in seen:
                    continue
                seen.append(handler.name)
                yield handler
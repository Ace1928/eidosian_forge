from __future__ import (absolute_import, division, print_function)
import os
import sys
import tempfile
import threading
import time
import typing as t
import multiprocessing.queues
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError
from ansible.executor.play_iterator import PlayIterator
from ansible.executor.stats import AggregateStats
from ansible.executor.task_result import TaskResult
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.playbook.play_context import PlayContext
from ansible.playbook.task import Task
from ansible.plugins.loader import callback_loader, strategy_loader, module_loader
from ansible.plugins.callback import CallbackBase
from ansible.template import Templar
from ansible.vars.hostvars import HostVars
from ansible.vars.reserved import warn_if_reserved
from ansible.utils.display import Display
from ansible.utils.lock import lock_decorator
from ansible.utils.multiprocessing import context as multiprocessing_context
from dataclasses import dataclass
def load_callbacks(self):
    """
        Loads all available callbacks, with the exception of those which
        utilize the CALLBACK_TYPE option. When CALLBACK_TYPE is set to 'stdout',
        only one such callback plugin will be loaded.
        """
    if self._callbacks_loaded:
        return
    stdout_callback_loaded = False
    if self._stdout_callback is None:
        self._stdout_callback = C.DEFAULT_STDOUT_CALLBACK
    if isinstance(self._stdout_callback, CallbackBase):
        stdout_callback_loaded = True
    elif isinstance(self._stdout_callback, string_types):
        if self._stdout_callback not in callback_loader:
            raise AnsibleError('Invalid callback for stdout specified: %s' % self._stdout_callback)
        else:
            self._stdout_callback = callback_loader.get(self._stdout_callback)
            self._stdout_callback.set_options()
            stdout_callback_loaded = True
    else:
        raise AnsibleError('callback must be an instance of CallbackBase or the name of a callback plugin')
    callback_list = list(callback_loader.all(class_only=True))
    for c in C.CALLBACKS_ENABLED:
        plugin = callback_loader.get(c, class_only=True)
        if plugin:
            if plugin not in callback_list:
                callback_list.append(plugin)
        else:
            display.warning("Skipping callback plugin '%s', unable to load" % c)
    for callback_plugin in callback_list:
        callback_type = getattr(callback_plugin, 'CALLBACK_TYPE', '')
        callback_needs_enabled = getattr(callback_plugin, 'CALLBACK_NEEDS_ENABLED', getattr(callback_plugin, 'CALLBACK_NEEDS_WHITELIST', False))
        cnames = getattr(callback_plugin, '_redirected_names', [])
        if cnames:
            callback_name = cnames[0]
        else:
            callback_name, ext = os.path.splitext(os.path.basename(callback_plugin._original_path))
        display.vvvvv("Attempting to use '%s' callback." % callback_name)
        if callback_type == 'stdout':
            if callback_name != self._stdout_callback or stdout_callback_loaded:
                display.vv("Skipping callback '%s', as we already have a stdout callback." % callback_name)
                continue
            stdout_callback_loaded = True
        elif callback_name == 'tree' and self._run_tree:
            pass
        elif not self._run_additional_callbacks or (callback_needs_enabled and (C.CALLBACKS_ENABLED is None or callback_name not in C.CALLBACKS_ENABLED)):
            continue
        try:
            callback_obj = callback_plugin()
            if callback_obj:
                if callback_obj not in self._callback_plugins:
                    callback_obj.set_options()
                    self._callback_plugins.append(callback_obj)
                else:
                    display.vv("Skipping callback '%s', already loaded as '%s'." % (callback_plugin, callback_name))
            else:
                display.warning("Skipping callback '%s', as it does not create a valid plugin instance." % callback_name)
                continue
        except Exception as e:
            display.warning("Skipping callback '%s', unable to load due to: %s" % (callback_name, to_native(e)))
            continue
    self._callbacks_loaded = True
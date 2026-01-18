import functools
from taskflow.engines.action_engine.actions import base
from taskflow import logging
from taskflow import states
from taskflow import task as task_atom
from taskflow.types import failure
def schedule_reversion(self, task):
    self.change_state(task, states.REVERTING, progress=0.0)
    arguments = self._storage.fetch_mapped_args(task.revert_rebind, atom_name=task.name, optional_args=task.revert_optional)
    task_uuid = self._storage.get_atom_uuid(task.name)
    task_result = self._storage.get(task.name)
    failures = self._storage.get_failures()
    if task.notifier.can_be_registered(task_atom.EVENT_UPDATE_PROGRESS):
        progress_callback = functools.partial(self._on_update_progress, task)
    else:
        progress_callback = None
    return self._task_executor.revert_task(task, task_uuid, arguments, task_result, failures, progress_callback=progress_callback)
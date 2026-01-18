import functools
from taskflow.engines.action_engine.actions import base
from taskflow import logging
from taskflow import states
from taskflow import task as task_atom
from taskflow.types import failure
def change_state(self, task, state, progress=None, result=base.Action.NO_RESULT):
    old_state = self._storage.get_atom_state(task.name)
    if self._is_identity_transition(old_state, state, task, progress=progress):
        return
    if state in self.SAVE_RESULT_STATES:
        save_result = None
        if result is not self.NO_RESULT:
            save_result = result
        self._storage.save(task.name, save_result, state)
    else:
        self._storage.set_atom_state(task.name, state)
    if progress is not None:
        self._storage.set_task_progress(task.name, progress)
    task_uuid = self._storage.get_atom_uuid(task.name)
    details = {'task_name': task.name, 'task_uuid': task_uuid, 'old_state': old_state}
    if result is not self.NO_RESULT:
        details['result'] = result
    self._notifier.notify(state, details)
    if progress is not None:
        task.update_progress(progress)
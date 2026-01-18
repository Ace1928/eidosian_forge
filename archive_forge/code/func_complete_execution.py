import functools
from taskflow.engines.action_engine.actions import base
from taskflow import logging
from taskflow import states
from taskflow import task as task_atom
from taskflow.types import failure
def complete_execution(self, task, result):
    if isinstance(result, failure.Failure):
        self.change_state(task, states.FAILURE, result=result)
    else:
        self.change_state(task, states.SUCCESS, result=result, progress=1.0)
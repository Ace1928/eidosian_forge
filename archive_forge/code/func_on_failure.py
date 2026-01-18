from taskflow.engines.action_engine.actions import base
from taskflow import retry as retry_atom
from taskflow import states
from taskflow.types import failure
def on_failure(self, retry, atom, last_failure):
    self._storage.save_retry_failure(retry.name, atom.name, last_failure)
    arguments = self._get_retry_args(retry)
    return retry.on_failure(**arguments)
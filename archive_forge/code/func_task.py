from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
@DeviceSpecV2.task.setter
def task(self, task):
    self._task = _as_int_or_none(task)
    self._as_string, self._hash = (None, None)
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
@DeviceSpecV2.job.setter
def job(self, job):
    self._job = _as_str_or_none(job)
    self._as_string, self._hash = (None, None)
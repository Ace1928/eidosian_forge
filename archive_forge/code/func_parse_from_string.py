from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
def parse_from_string(self, spec):
    self.job, self.replica, self.task, self.device_type, self.device_index = self._string_to_components(spec)
    return self
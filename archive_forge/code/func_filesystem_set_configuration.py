from tensorflow.python.ops import gen_filesystem_ops as _gen_filesystem_ops
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.filesystem_set_configuration')
def filesystem_set_configuration(scheme, key, value, name=None):
    """Set configuration of the file system.

  Args:
    scheme: File system scheme.
    key: The name of the configuration option.
    value: The value of the configuration option.
    name: A name for the operation (optional).

  Returns:
    None.
  """
    return _gen_filesystem_ops.file_system_set_configuration(scheme, key=key, value=value, name=name)
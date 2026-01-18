from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
def canonicalize(d, default=None):
    """Canonicalize device string.

  If d has missing components, the rest would be deduced from the `default`
  argument or from '/replica:0/task:0/device:CPU:0'. For example:
    If d = '/cpu:0', default='/job:worker/task:1', it returns
      '/job:worker/replica:0/task:1/device:CPU:0'.
    If d = '/cpu:0', default='/job:worker', it returns
      '/job:worker/replica:0/task:0/device:CPU:0'.
    If d = '/gpu:0', default=None, it returns
      '/replica:0/task:0/device:GPU:0'.

  Note: This uses "job:localhost" as the default if executing eagerly.

  Args:
    d: a device string or tf.config.LogicalDevice
    default: a string for default device if d doesn't have all components.

  Returns:
    a canonicalized device string.
  """
    if isinstance(d, context.LogicalDevice):
        d = tf_device.DeviceSpec.from_string(d.name)
    else:
        d = tf_device.DeviceSpec.from_string(d)
    assert d.device_type is None or d.device_type == d.device_type.upper(), "Device type '%s' must be all-caps." % (d.device_type,)
    result = tf_device.DeviceSpec(replica=0, task=0, device_type='CPU', device_index=0)
    if ops.executing_eagerly_outside_functions():
        host_cpu = tf_device.DeviceSpec.from_string(config.list_logical_devices('CPU')[0].name)
        if host_cpu.job:
            result = result.make_merged_spec(host_cpu)
        else:
            result = result.replace(job='localhost')
    if default:
        result = result.make_merged_spec(tf_device.DeviceSpec.from_string(default))
    result = result.make_merged_spec(d)
    return result.to_string()
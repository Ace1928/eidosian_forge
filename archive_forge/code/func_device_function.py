from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util.tf_export import tf_export
def device_function(self, op):
    """Choose a device for `op`.

    Args:
      op: an `Operation`.

    Returns:
      The device to use for the `Operation`.
    """
    if not self._merge_devices and op.device:
        return op.device
    current_device = pydev.DeviceSpec.from_string(op.device or '')
    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
    if self._ps_tasks and self._ps_device and (node_def.op in self._ps_ops):
        ps_device = pydev.DeviceSpec.from_string(self._ps_device)
        current_job, ps_job = (current_device.job, ps_device.job)
        if ps_job and (not current_job or current_job == ps_job):
            ps_device = ps_device.replace(task=self._ps_strategy(op))
        ps_device = ps_device.make_merged_spec(current_device)
        return ps_device.to_string()
    worker_device = pydev.DeviceSpec.from_string(self._worker_device or '')
    worker_device = worker_device.make_merged_spec(current_device)
    return worker_device.to_string()
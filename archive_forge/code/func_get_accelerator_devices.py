import abc
import collections
import six
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def get_accelerator_devices(master, config_proto):
    """Returns accelerator devices given a master and a configuration."""
    if context.executing_eagerly():
        logical_devices = config.list_logical_devices()
        devices = []
        for d in logical_devices:
            if d.device_type == 'CPU' or d.device_type == 'XLA_CPU':
                continue
            devices.append(session._DeviceAttributes(d.name, d.device_type, 0, 0))
        return devices
    else:
        with ops.Graph().as_default():
            with session.Session(master, config=config_proto) as s:
                devices = s.list_devices()
        return devices
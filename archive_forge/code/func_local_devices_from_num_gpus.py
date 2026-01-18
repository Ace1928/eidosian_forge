from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
def local_devices_from_num_gpus(num_gpus):
    """Returns device strings for local GPUs or CPU."""
    return tuple(('/device:GPU:%d' % i for i in range(num_gpus))) or ('/device:CPU:0',)
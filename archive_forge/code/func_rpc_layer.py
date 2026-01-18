import abc
import collections
import six
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
@rpc_layer.setter
def rpc_layer(self, rpc_layer):
    self._rpc_layer = rpc_layer
import abc
import collections
import six
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def format_master_url(master, rpc_layer=None):
    if rpc_layer:
        return '%s://%s' % (rpc_layer, master)
    else:
        return master
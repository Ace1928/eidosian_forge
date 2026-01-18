import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def get_num_gpus():
    """Returns the number of GPUs visible on the current node.

  Currently only implemented for NVIDIA GPUs.
  """
    return _get_num_nvidia_gpus()
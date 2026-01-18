import tempfile
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
def make_test_cluster(self, *args, **kwargs):
    if 'data_transfer_protocol' not in kwargs:
        kwargs['data_transfer_protocol'] = self.default_data_transfer_protocol
    return TestCluster(*args, **kwargs)
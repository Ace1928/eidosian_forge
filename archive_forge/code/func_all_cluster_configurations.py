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
def all_cluster_configurations():
    with_work_dir = combinations.combine(work_dir=TMP_WORK_DIR, fault_tolerant_mode=[True, False])
    without_work_dir = combinations.combine(work_dir=NO_WORK_DIR, fault_tolerant_mode=False)
    return with_work_dir + without_work_dir
import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
@staticmethod
def float_operation():
    """Options used to profile float operations.

    Please see https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/profile_model_architecture.md
    on the caveats of calculating float operations.

    Returns:
      A dict of profiling options.
    """
    return {'max_depth': 10000, 'min_bytes': 0, 'min_micros': 0, 'min_params': 0, 'min_float_ops': 1, 'min_occurrence': 0, 'order_by': 'float_ops', 'account_type_regexes': ['.*'], 'start_name_regexes': ['.*'], 'trim_name_regexes': [], 'show_name_regexes': ['.*'], 'hide_name_regexes': [], 'account_displayed_op_only': True, 'select': ['float_ops'], 'step': -1, 'output': 'stdout'}
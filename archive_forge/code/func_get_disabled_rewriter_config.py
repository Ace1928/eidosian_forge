import functools
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def get_disabled_rewriter_config():
    global _rewriter_config_optimizer_disabled
    if _rewriter_config_optimizer_disabled is None:
        config = config_pb2.ConfigProto()
        rewriter_config = config.graph_options.rewrite_options
        rewriter_config.disable_meta_optimizer = True
        _rewriter_config_optimizer_disabled = config.SerializeToString()
    return _rewriter_config_optimizer_disabled
import collections
import os
import re
from packaging import version
from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes
def disable_non_trt_optimizers_in_rewriter_config(rewriter_config):
    """Modifies rewriter_config to disable all non-TRT optimizations."""
    off = rewriter_config_pb2.RewriterConfig.OFF
    rewriter_config.arithmetic_optimization = off
    rewriter_config.auto_mixed_precision = off
    rewriter_config.auto_parallel.enable = False
    rewriter_config.constant_folding = off
    rewriter_config.debug_stripper = off
    rewriter_config.dependency_optimization = off
    rewriter_config.disable_meta_optimizer = False
    rewriter_config.disable_model_pruning = True
    rewriter_config.function_optimization = off
    rewriter_config.implementation_selector = off
    rewriter_config.layout_optimizer = off
    rewriter_config.loop_optimization = off
    rewriter_config.memory_optimization = rewriter_config_pb2.RewriterConfig.NO_MEM_OPT
    rewriter_config.min_graph_nodes = -1
    rewriter_config.pin_to_host_optimization = off
    rewriter_config.remapping = off
    rewriter_config.scoped_allocator_optimization = off
    rewriter_config.shape_optimization = off
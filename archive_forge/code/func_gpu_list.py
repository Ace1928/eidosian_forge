from dataclasses import dataclass, field
from typing import Tuple
from ..utils import cached_property, is_tf_available, logging, requires_backends
from .benchmark_args_utils import BenchmarkArguments
@property
def gpu_list(self):
    requires_backends(self, ['tf'])
    return tf.config.list_physical_devices('GPU')
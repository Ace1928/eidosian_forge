from dataclasses import dataclass, field
from typing import Tuple
from ..utils import cached_property, is_torch_available, is_torch_tpu_available, logging, requires_backends
from .benchmark_args_utils import BenchmarkArguments
@property
def n_gpu(self):
    requires_backends(self, ['torch'])
    return self._setup_devices[1]
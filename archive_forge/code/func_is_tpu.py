from dataclasses import dataclass, field
from typing import Tuple
from ..utils import cached_property, is_torch_available, is_torch_tpu_available, logging, requires_backends
from .benchmark_args_utils import BenchmarkArguments
@property
def is_tpu(self):
    return is_torch_tpu_available() and self.tpu
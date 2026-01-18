import gzip
import json
import os
import tempfile
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from warnings import warn
import torch
import torch.autograd.profiler as prof
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import (
from torch.autograd import kineto_available, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline
def start_trace(self):
    assert self.profiler is not None
    self.profiler._start_trace()
    if self.profile_memory:
        self.add_metadata_json('profile_memory', '1')
    if self.with_stack:
        self.add_metadata_json('with_stack', '1')
    if self.record_shapes:
        self.add_metadata_json('record_shapes', '1')
    if self.with_modules:
        self.add_metadata_json('with_modules', '1')
    if self.with_flops:
        self.add_metadata_json('with_flops', '1')
    if kineto_available():
        dist_info = self._get_distributed_info()
        if dist_info:
            self.add_metadata_json('distributedInfo', json.dumps(dist_info))
        if hasattr(torch, '_inductor'):
            import torch._inductor.config as inductor_config
            if inductor_config.triton.cudagraphs:
                os.environ['DISABLE_CUPTI_LAZY_REINIT'] = '1'
                self.add_metadata_json('DISABLE_CUPTI_LAZY_REINIT', '1')
                os.environ['TEARDOWN_CUPTI'] = '0'
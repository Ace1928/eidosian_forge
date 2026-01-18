from collections import defaultdict
from typing import Any, Dict, List, Optional
from warnings import warn
import torch
import torch.cuda
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import _ExperimentalConfig
from torch.autograd import (
from torch.autograd.profiler_util import (
from torch.futures import Future
def createFunctionEventForMemoryEvents(evt):
    rel_start_us = evt.start_us() - trace_start_us
    fe = FunctionEvent(id=max_evt_id, name=evt.name(), trace_name=None, thread=evt.start_thread_id(), start_us=rel_start_us, end_us=rel_start_us, fwd_thread=evt.start_thread_id(), input_shapes=[], stack=[], scope=0, use_device=self.use_device, cpu_memory_usage=_cpu_memory_usage(evt), cuda_memory_usage=_cuda_memory_usage(evt), privateuse1_memory_usage=_privateuse1_memory_usage(evt), is_async=False, sequence_nr=-1, device_type=DeviceType.CPU, device_index=0)
    return fe
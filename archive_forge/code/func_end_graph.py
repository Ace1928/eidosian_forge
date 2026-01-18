import builtins
import copy
import functools
import hashlib
import inspect
import json
import logging
import math
import operator
import os
import os.path
import re
import threading
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import torch
import torch.autograd.profiler as autograd_profiler
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import dynamo_timed
from torch.utils._triton import has_triton, has_triton_package
from . import config
from .codecache import cache_dir, CudaKernelParamCache
from .coordinate_descent_tuner import CoordescTuner
from .ir import ReductionHint, TileHint
from .utils import (
def end_graph():
    if len(collected_calls) == 0:
        return
    overall_time = sum((call[0] for call in collected_calls))
    overall_gb = sum((call[1] for call in collected_calls))
    cur_file = inspect.stack()[1].filename
    summary_str = f'SUMMARY ({cur_file})\n{overall_time:.2f}ms   \t {overall_gb:.2f} GB\t {overall_gb / (overall_time / 1000.0):.2f}GB/s'
    print(summary_str)
    print()
    output_file = config.profile_bandwidth_output
    if output_file is not None:
        sorted_calls = sorted(collected_calls, key=lambda c: float(c[0]), reverse=True)
        try:
            with open(output_file, 'a') as file:
                log.debug('Save profile bandwidth results to %s', output_file)
                file.write('====================\n')
                file.write(f'TRITON KERNELS BANDWIDTH INFO ({cur_file})\n')
                for ms, num_gb, gb_per_s, kernel_name in sorted_calls:
                    percentage = f'{ms / overall_time * 100:.2f}%'
                    suffix = f' \t {percentage} \t {kernel_name}'
                    bw_info_str = create_bandwidth_info_str(ms, num_gb, gb_per_s, suffix=suffix)
                    file.write(bw_info_str + '\n')
                file.write(f'{summary_str}\n\n')
        except Exception as e:
            log.warning('failed to write profile bandwidth result into %s: %s', output_file, e)
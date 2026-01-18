import functools
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, List
from . import language as tl
from ._C.libtriton.triton import runtime
@contextmanager
def set_gpu_clock(ref_sm_clock=1350, ref_mem_clock=1215):
    try:
        subprocess.check_output(['nvidia-smi', '-i', '0', '-pm', '1'])
        subprocess.check_output(['nvidia-smi', '-i', '0', f'--lock-gpu-clocks={ref_sm_clock},{ref_sm_clock}'])
        subprocess.check_output(['nvidia-smi', '-i', '0', f'--lock-memory-clocks={ref_mem_clock},{ref_mem_clock}'])
        cur_sm_clock = nvsmi(['clocks.current.sm'])[0]
        cur_mem_clock = nvsmi(['clocks.current.memory'])[0]
        assert abs(cur_sm_clock - ref_sm_clock) < 10, f'GPU SMs must run at {ref_sm_clock} MHz'
        assert abs(cur_mem_clock - ref_mem_clock) < 10, f'GPU SMs must run at {ref_mem_clock} MHz'
        tflops = 1e-06 * 2 * 108 * 4 * 256 * ref_sm_clock
        gbps = 640 * 2 * ref_mem_clock * 0.001
        yield (tflops, gbps)
    finally:
        subprocess.check_output(['nvidia-smi', '-i', '0', '-pm', '0'])
        subprocess.check_output(['nvidia-smi', '-i', '0', '-rgc'])
        subprocess.check_output(['nvidia-smi', '-i', '0', '-rmc'])
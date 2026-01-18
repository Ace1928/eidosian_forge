import pickle
import sys
import time
import torch
import torch.utils.benchmark as benchmark_utils
def extra_overhead(self, result):
    numel = int(result.numel())
    if numel > 5000:
        time.sleep(numel * self._extra_ns_per_element * 1e-09)
    return result
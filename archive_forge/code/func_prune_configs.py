from __future__ import annotations
import builtins
import time
from typing import Dict
from ..testing import do_bench
from .jit import KernelInterface
def prune_configs(self, kwargs):
    pruned_configs = self.configs
    if self.early_config_prune:
        pruned_configs = self.early_config_prune(self.configs, self.nargs)
    if self.perf_model:
        top_k = self.configs_top_k
        if isinstance(top_k, float) and top_k <= 1.0:
            top_k = int(len(self.configs) * top_k)
        if len(pruned_configs) > top_k:
            est_timing = {config: self.perf_model(**self.nargs, **kwargs, **config.kwargs, num_stages=config.num_stages, num_warps=config.num_warps, num_ctas=config.num_ctas, enable_warp_specialization=config.enable_warp_specialization, enable_persistent=config.enable_persistent) for config in pruned_configs}
            pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
    return pruned_configs
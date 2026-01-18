import gc
import math
from collections import namedtuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
import triton
from triton.ops.blocksparse import matmul as blocksparse_matmul
from xformers.benchmarks.utils import pretty_barplot
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import SparseCS, _matmul_with_mask
def bench_all(self, a, b, tests, mask_config, sparsity, baseline_name, op_flops, dict_key):
    if self.do_accuracy_check:
        self.check_all(tests, a, b)
    for testcase in tests:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            fn = testcase.prepare_callable(a, b, testcase.mask, testcase.config)
            ms = triton.testing.do_bench(fn)[0]
            flops = op_flops / ms * 1000.0
            mem = self.do_mem(fn)
        except Exception:
            ms = -1
            flops = -1
            mem = -1
        self.add_kv(self.results['time'], dict_key, ms, testcase)
        self.add_kv(self.results['flops'], dict_key, flops, testcase)
        self.add_kv(self.results['memory'], dict_key, mem, testcase)
        speedup = self.results['time'][dict_key][baseline_name] / ms
        memory_savings = self.results['memory'][dict_key][baseline_name] / mem
        self.add_kv(self.results['speedup'], dict_key, speedup, testcase)
        self.add_kv(self.results['flops'], dict_key, flops, testcase)
        self.add_kv(self.results['memory_savings'], dict_key, memory_savings, testcase)
        desc = f'sparsity={sparsity}, ops={op_flops}, time={ms}, tflops={flops}, mem={mem}'
        print(f'{testcase.name} --> {mask_config}, {desc}')
import argparse
import itertools as it
import multiprocessing
import multiprocessing.dummy
import os
import pickle
import queue
import subprocess
import tempfile
import textwrap
import numpy as np
import torch
from torch.utils.benchmark.op_fuzzers import unary
from torch.utils.benchmark import Timer, Measurement
from typing import Dict, Tuple, List
def construct_table(results, device_str, test_variance):
    device_str = f'== {device_str} {(' (Variance Test)' if test_variance else '')}  '.ljust(40, '=')
    print(f'{'=' * 40}\n{device_str}\n{'=' * 40}\n')
    results = sorted(((key, (r_ref, r_pr), r_pr.median / r_ref.median - 1) for key, (r_ref, r_pr) in results), key=lambda i: i[2])
    n = len(results)
    n_regressed = len([i for i in results if i[2] > 0.05])
    n_improved = len([i for i in results if i[2] < -0.05])
    n_unchanged = n - n_improved - n_regressed
    legends = ['Improved  (>5%):', 'Regressed (>5%):', 'Within 5%:']
    for legend, count in zip(legends, [n_improved, n_regressed, n_unchanged]):
        print(f'{legend:<17} {count:>6}  ({count / len(results) * 100:>3.0f}%)')
    keys_to_print = {i[0] for i in results[20:30]} | {i[0] for i in results[int(n // 2 - 5):int(n // 2 + 5)]} | {i[0] for i in results[-30:-20]}
    ellipsis_after = {results[29][0], results[int(n // 2 + 4)][0]}
    column_labels = f'Relative Δ     Absolute Δ      |      numel{'':>8}dtype{'':>14}shape{'':>10}steps{'':>10}layout{'':>7}task specific\n{'=' * 126}'
    _, result_log_file = tempfile.mkstemp(suffix='.log')
    with open(result_log_file, 'w') as f:
        f.write(f'{device_str}\n\n{column_labels}\n')
        print(f'\n{column_labels}\n[First twenty omitted (these tend to be noisy) ]')
        for key, (r_ref, r_pr), rel_diff in results:
            row = row_str(rel_diff, r_pr.median - r_ref.median, r_ref)
            f.write(f'{row}\n')
            if key in keys_to_print:
                print(row)
            if key in ellipsis_after:
                print('...')
        print('[Last twenty omitted (these tend to be noisy) ]')
    print(textwrap.dedent('\n        steps:\n            Indicates that `x` is sliced from a larger Tensor. For instance, if\n            shape is [12, 4] and steps are [2, 1], then a larger Tensor of size\n            [24, 4] was created, and then x = base_tensor[::2, ::1]. Omitted if\n            all elements are ones.\n\n        layout:\n            Indicates that `x` is not contiguous due to permutation. Invoking\n            `x.permute(layout)` (e.g. x.permute((2, 0, 1)) if layout = [2, 0, 1])\n            would produce a Tensor with physical memory layout matching logical\n            memory layout. (Though still not contiguous if `steps` contains\n            non-one elements.)\n        '))
    print(f'\nComplete results in: {result_log_file}')
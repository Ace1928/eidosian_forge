import argparse
import contextlib
import copy
import csv
import functools
import glob
import itertools
import logging
import math
import os
import tempfile
from collections import defaultdict, namedtuple
from dataclasses import replace
from typing import Any, Dict, Generator, Iterator, List, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from torch.utils import benchmark
def bench_functions(test_cases: List[TestCase], shapes, metric_transform, unit, title=''):
    device = torch.device('cuda')
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        results: Dict[str, Any] = {}
        for B, M, K in shapes:
            a = torch.rand(B, M, K, device=device, dtype=dtype, requires_grad=True)
            for testcase in test_cases:
                time = triton.testing.do_bench(lambda: testcase.function(a))[0]
                metric = metric_transform(a, time)
                key = f'B={B}, M={M}, K={K}'
                if key not in results:
                    results[key] = {}
                results[key][testcase.name] = f'{metric:.1f}'
        pretty_print(results, title=' ------------- Type: {} ------------- '.format(dtype), units=unit)
        pretty_plot(results, title + str(dtype), unit, dash_key='pytorch')
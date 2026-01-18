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
def benchmark_main_helper2(name: str, functions, fw: bool=False, bw: bool=False, cuda_graph: bool=True, **kwargs) -> None:
    assert fw or bw

    def handle_case(**case) -> Iterator[benchmark.Timer]:
        for k, benchmark_cls in functions.items():
            benchmark_object = benchmark_cls(**case, bw=bw)
            label = benchmark_object.label
            label += 'fw' if fw else ''
            label += 'bw' if bw else ''

            def run_one():
                if fw:
                    benchmark_object.fw()
                if bw:
                    benchmark_object.bw()
            if cuda_graph:
                run_one()
                benchmark_object = benchmark_cls(**case, bw=bw)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    run_one()

                def run_one():
                    g.replay()
            yield benchmark.Timer(stmt='fn()', globals={'fn': run_one}, label=label, description=k, sub_label=benchmark_object.sub_label)
    handle_case.__name__ = name
    benchmark_main_helper(handle_case, **kwargs)
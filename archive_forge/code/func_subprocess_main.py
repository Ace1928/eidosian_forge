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
def subprocess_main(args):
    seed = args.DETAIL_seed
    cuda = args.DETAIL_device == _GPU
    with open(args.DETAIL_result_file, 'ab') as f:
        for dtype_str in _DTYPES_TO_TEST[args.pr]:
            dtype = _DTYPE_STR_TO_DTYPE[dtype_str]
            iterator = unary.UnaryOpFuzzer(seed=seed, dtype=dtype, cuda=cuda).take(_RUNS_PER_LOOP)
            for i, (tensors, tensor_parameters, params) in enumerate(iterator):
                params['dtype_str'] = dtype_str
                stmt, label = construct_stmt_and_label(args.pr, params)
                timer = Timer(stmt=stmt, globals=tensors, label=label, description=f'[{i}, seed={seed}] ({dtype_str}), stmt = {stmt}', env=args.DETAIL_env)
                measurement = timer.blocked_autorange(min_run_time=_MIN_RUN_SEC)
                measurement.metadata = {'tensor_parameters': tensor_parameters, 'params': params}
                print(measurement)
                pickle.dump(measurement, f)
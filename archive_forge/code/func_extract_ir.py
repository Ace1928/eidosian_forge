from contextlib import contextmanager
from typing import Any, List, Tuple, cast
import random
import torch
import time
from torch.utils.benchmark import Timer
def extract_ir(filename: str) -> List[str]:
    BEGIN = '<GRAPH_EXPORT>'
    END = '</GRAPH_EXPORT>'
    pfx = None
    current = ''
    graphs = []
    with open(filename) as f:
        split_strs = f.read().split(BEGIN)
        for i, split_str in enumerate(split_strs):
            if i == 0:
                continue
            end_loc = split_str.find(END)
            if end_loc == -1:
                continue
            s = split_str[:end_loc]
            pfx = split_strs[i - 1].splitlines()[-1]
            lines = [x[len(pfx):] for x in s.splitlines(keepends=True)]
            graphs.append(''.join(lines))
    return graphs
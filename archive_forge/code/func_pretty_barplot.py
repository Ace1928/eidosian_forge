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
def pretty_barplot(results, title, units: str, filename=None, dash_key=''):
    """Graph out the contents of a dict.
    Dash key means that if the result label has this key, then it will be displayed with a dash
    """
    if not filename:
        filename = title + '.png'
    filename = filename.replace(' ', '_').replace('/', '_').replace('-', '_').replace(':', '')
    xlabels = list(results.keys())
    workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
    for v in results.values():
        for k in v.keys():
            workloads[k].append(float(v[k]))
    options = list(workloads.keys())
    group_len = len(options)
    for key in workloads.keys():
        num_groups = len(workloads[key])
        break
    group_width = group_len + 1
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)
    for idx in range(group_len):
        option = options[idx]
        values = workloads[option]
        xloc = np.arange(1 + idx, group_width * num_groups, group_width)
        plt.bar(xloc, values, width=1, edgecolor='black')
    plt.title(title)
    plt.legend(list(workloads.keys()), loc='upper right')
    plt.ylabel(units)
    ax = plt.gca()
    xticks_loc = np.arange(1 + (group_len - 1) / 2.0, group_width * num_groups, group_width)
    ax.set_xticks(xticks_loc, xlabels)
    plt.xticks(rotation=45)
    plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.savefig(filename, bbox_inches='tight')
    plt.close(f)
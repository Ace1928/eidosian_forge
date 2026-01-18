import glob
import os
import os.path as osp
import sys
import re
import copy
import time
import math
import logging
import itertools
from ast import literal_eval
from collections import defaultdict
from argparse import ArgumentParser, ArgumentError, REMAINDER, RawTextHelpFormatter
import importlib
import memory_profiler as mp
def filter_mprofile_mem_usage_by_function(prof, func):
    if func is None:
        return prof['mem_usage']
    if func not in prof['func_timestamp']:
        raise ValueError(str(func) + ' was not found.')
    time_ranges = prof['func_timestamp'][func]
    filtered_memory = []
    for mib, ts in zip(prof['mem_usage'], prof['timestamp']):
        for rng in time_ranges:
            if rng[0] <= ts <= rng[1]:
                filtered_memory.append(mib)
    return filtered_memory
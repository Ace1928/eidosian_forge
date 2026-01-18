import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from multiprocessing import cpu_count
from typing import (
import pandas as pd
from tqdm.auto import tqdm
from cmdstanpy import (
from cmdstanpy.cmdstan_args import (
from cmdstanpy.stanfit import (
from cmdstanpy.utils import (
from cmdstanpy.utils.filesystem import temp_inits, temp_single_json
from . import progress as progbar
def progress_hook(line: str, idx: int) -> None:
    if line == 'Done':
        for pbar in pbars.values():
            pbar.postfix[0]['value'] = 'Sampling completed'
            pbar.update(total - pbar.n)
            pbar.close()
    else:
        match = pat.match(line)
        if match:
            idx = int(match.group(1))
            mline = match.group(2).strip()
        elif line.startswith('Iteration'):
            mline = line
            idx = chain_ids[idx]
        else:
            return
        if 'Sampling' in mline:
            pbars[idx].colour = 'blue'
        pbars[idx].update(1)
        pbars[idx].postfix[0]['value'] = mline
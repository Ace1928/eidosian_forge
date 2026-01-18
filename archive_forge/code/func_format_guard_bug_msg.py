import collections
from contextlib import contextmanager
from typing import List, Tuple
import torch
import torch.fx.traceback as fx_traceback
def format_guard_bug_msg(aot_config, expected):
    return f'At compilation time, graph {aot_config.aot_id} was compiled under the assumption that {expected}, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.'
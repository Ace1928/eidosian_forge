from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from fairscale.nn import FullyShardedDataParallel
def all_gathered_memory(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
    gathered_memory = [t.all_gathered for t in memory_traces]
    cumul_gathered_memory = [t.cumul_all_gathered for t in memory_traces]
    x, y_forward, y_backward = self._split_forward_backward(memory_traces, gathered_memory)
    ax.plot(x, y_forward, x, y_backward, label=job_name)
    ax.plot(x, cumul_gathered_memory, label=job_name)
    self._y_axis_in_gigabytes(ax)
    max_index = np.argmax(cumul_gathered_memory)
    max_trace = memory_traces[max_index]
    max_module = '.'.join([n for n in max_trace.module_name.split('.') if not n.startswith('_')])
    ax.set_ylim([None, max_trace.cumul_all_gathered * 1.1])
    x_text, y_text = (max(0, max_index * 0.8), max_trace.cumul_all_gathered * 1.04)
    ax.text(x_text, y_text, f'{max_module} (fwd)', fontdict=self.font)
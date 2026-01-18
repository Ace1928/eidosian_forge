import logging
import os
import queue
import socket
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.nn as nn
import torch.profiler
import torch.utils.hooks
def format_summary(self) -> str:
    if len(self.summary) == 0:
        return ''
    pad_titles = max((len(title) for title, value in self.summary))
    return 'summary:\n' + '\n'.join([f'  {title.ljust(pad_titles)}: {value}' for title, value in self.summary])
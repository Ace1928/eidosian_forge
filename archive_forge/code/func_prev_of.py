import json
import math
import os
import re
from typing import Dict, List, Optional, Set
import torch
import torch.utils.benchmark as benchmark
from torch._C._profiler import (
from torch.profiler import profile
from torch.profiler._utils import index_of_first_match, traverse_bfs, traverse_dfs
def prev_of(self, event: _ProfilerEvent):
    prev_events, _ = self.siblings_of(event)
    return prev_events[-1] if prev_events else None
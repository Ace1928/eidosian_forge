from collections import defaultdict
from itertools import chain
import pickle
from typing import (
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torch.utils._python_dispatch import TorchDispatchMode
def save_stats(self, path: str) -> None:
    """Save the stats using pickle during runtime if users want to plot the traces in other places like notebook."""
    stats = {'memories_allocated': self.memories_allocated, 'memories_active': self.memories_active, 'memories_reserved': self.memories_reserved, 'markers': self._markers, 'num_alloc_retries': self._num_cuda_retries}
    with open(path, 'wb') as f:
        pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
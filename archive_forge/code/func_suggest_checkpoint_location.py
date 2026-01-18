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
def suggest_checkpoint_location(traces: List[LayerMemoryTrace], num_checkpoints: int, num_skipped_layers: int=0) -> SuggestedCheckpoints:
    """
    Given a trace of a model, collected with or without checkpoint,
    return the best places to insert a reset of activation memory.

    The names of the returned modules are the boundaries of the
    suggested checkpoint_wrapper wrappings
    """
    visited = set()
    modules, allocations = ([], [])
    for t in traces:
        if t.is_forward:
            name = t.module_name
            memory = t.event.memory_activations
            if name not in visited:
                visited.add(name)
                modules.append(name)
                allocations.append(memory)
    if num_skipped_layers:
        modules = modules[num_skipped_layers:]
        allocations = allocations[num_skipped_layers:]
    max_memory, reset_indices = find_best_reset_points(allocations, num_checkpoints=num_checkpoints)
    return SuggestedCheckpoints(max_memory=max_memory, split_modules=[modules[i] for i in reset_indices], all_modules=modules)
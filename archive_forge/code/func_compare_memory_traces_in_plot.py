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
def compare_memory_traces_in_plot(memory_traces_by_job: Dict[str, List[LayerMemoryTrace]], figsize: Tuple[int, int]=(16, 20), capture: bool=False) -> Optional[Any]:
    """
    Create a plot of the memory allocation over time during the forward/backward
    passes, with a breakdown of the memory used for activation VS parameters
    """
    _assert_visualisation_library_installed()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=3)
    graph_creator = _MemoryGraphCreator()
    ax[0, 0].set_title('memory allocated')
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.allocated_memory_curve(ax[0, 0], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[0, 0].legend()
    ax[0, 1].set_title('memory reserved')
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.reserved_memory_curve(ax[0, 1], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[0, 1].legend()
    ax[1, 0].set_title('activation allocations')
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.activation_allocations(ax[1, 0], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[1, 0].legend()
    ax[1, 1].set_title('cumulative forward activations')
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.cumulative_activations(ax[1, 1], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[1, 1].legend()
    ax[2, 0].set_title('all gathered memory')
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.all_gathered_memory(ax[2, 0], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[2, 0].legend()
    ax[2, 1].set_title('parameter memory')
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.module_parameters(ax[2, 1], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[2, 1].legend()
    if not capture:
        plt.show()
        return None
    else:
        return matplotlib_figure_to_image(fig)
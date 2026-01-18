import inspect
import warnings
from typing import Any, List, Optional, Set
import torch
from torch.utils.data.datapipes.iter.sharding import (
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps
def apply_shuffle_settings(datapipe: DataPipe, shuffle: Optional[bool]=None) -> DataPipe:
    """
    Traverse the graph of ``DataPipes`` to find and set shuffle attribute.

    Apply the method to each `DataPipe` that has APIs of ``set_shuffle``
    and ``set_seed``.

    Args:
        datapipe: DataPipe that needs to set shuffle attribute
        shuffle: Shuffle option (default: ``None`` and no-op to the graph)
    """
    if shuffle is None:
        return datapipe
    graph = traverse_dps(datapipe)
    all_pipes = get_all_graph_pipes(graph)
    shufflers = [pipe for pipe in all_pipes if _is_shuffle_datapipe(pipe)]
    if not shufflers and shuffle:
        warnings.warn('`shuffle=True` was set, but the datapipe does not contain a `Shuffler`. Adding one at the end. Be aware that the default buffer size might not be sufficient for your task.')
        datapipe = datapipe.shuffle()
        shufflers = [datapipe]
    for shuffler in shufflers:
        shuffler.set_shuffle(shuffle)
    return datapipe
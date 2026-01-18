import functools
import pickle
from typing import Dict, Callable, Optional, TypeVar, Generic, Iterator
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.utils.common import (
from torch.utils.data.dataset import Dataset, IterableDataset
@classmethod
def set_reduce_ex_hook(cls, hook_fn):
    if MapDataPipe.reduce_ex_hook is not None and hook_fn is not None:
        raise Exception('Attempt to override existing reduce_ex_hook')
    MapDataPipe.reduce_ex_hook = hook_fn
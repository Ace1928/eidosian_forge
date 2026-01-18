import functools
import pickle
from typing import Dict, Callable, Optional, TypeVar, Generic, Iterator
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.utils.common import (
from torch.utils.data.dataset import Dataset, IterableDataset
@classmethod
def set_getstate_hook(cls, hook_fn):
    if MapDataPipe.getstate_hook is not None and hook_fn is not None:
        raise Exception('Attempt to override existing getstate_hook')
    MapDataPipe.getstate_hook = hook_fn
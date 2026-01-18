import io
import pickle
import warnings
from collections.abc import Collection
from typing import Dict, List, Optional, Set, Tuple, Type, Union
from torch.utils.data import IterDataPipe, MapDataPipe
from torch.utils.data._utils.serialization import DILL_AVAILABLE
def getstate_hook(ori_state):
    state = None
    if isinstance(ori_state, dict):
        state = {}
        for k, v in ori_state.items():
            if isinstance(v, (IterDataPipe, MapDataPipe, Collection)):
                state[k] = v
    elif isinstance(ori_state, (tuple, list)):
        state = []
        for v in ori_state:
            if isinstance(v, (IterDataPipe, MapDataPipe, Collection)):
                state.append(v)
    elif isinstance(ori_state, (IterDataPipe, MapDataPipe, Collection)):
        state = ori_state
    return state
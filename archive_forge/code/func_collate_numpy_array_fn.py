import collections
import contextlib
import re
import torch
from typing import Callable, Dict, Optional, Tuple, Type, Union
def collate_numpy_array_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]]=None):
    elem = batch[0]
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))
    return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)
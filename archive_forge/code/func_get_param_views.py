from contextlib import contextmanager
from itertools import chain
import typing
from typing import (
import torch
from torch import Tensor
import torch.nn as nn
from fairscale.internal.state_dict import replace_by_prefix_
def get_param_views(self, external_data_list: Optional[List[Optional[Tensor]]]=None) -> Iterator[Tensor]:
    """Used to get a generator over all views from a list of external data list."""
    params = self.flat_params
    if external_data_list is None:
        external_data_list = [None] * len(params)
    assert len(external_data_list) == len(params), f'Incorrect external data list: {len(external_data_list)} vs. {len(params)}'
    gens = []
    for p, data in zip(params, external_data_list):
        gens.append(p.get_param_views(data))
    return chain(*gens)
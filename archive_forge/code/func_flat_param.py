from contextlib import contextmanager
from itertools import chain
import typing
from typing import (
import torch
from torch import Tensor
import torch.nn as nn
from fairscale.internal.state_dict import replace_by_prefix_
@property
def flat_param(self) -> nn.Parameter:
    """We used to support only a single flat_param. This allows us to
        be backward compatible.
        """
    assert len(self.flat_params) == 1, f'Incorrect access to flat_param: len(self.flat_params)={len(self.flat_params)}'
    return self.flat_params[0]
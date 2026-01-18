import torch
from torch import Tensor
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils import _pytree as pytree
from functools import partial
from torch.utils._mode_utils import no_dispatch, all_same_mode
import torch.autograd.forward_ad as fwAD
from typing import Callable
import re
def is_view_fn(func):
    return func.overloadpacket.__name__ in {'as_strided', 'detach', 'diagonal', 'expand', 'expand_as', 'movedim', 'narrow', 'permute', 'select', 'squeeze', 'transpose', 't', 'real', 'imag', 'view_as_real', 'view_as_complex', 'unflatten', 'unfold', 'unsqueeze', 'view', 'view_as', 'unbind', 'split', 'split_with_sizes', 'vsplit', 'hsplit', 'tensor_split', 'chunk', 'swapaxes', 'slice', '_reshape_alias', '_unsafe_view', '_conj', 'alias'}
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
def raise_composite_compliance_error(err, additional_info=''):
    raise RuntimeError(f'Composite compliance check failed with the above error.\n{additional_info}If you are adding an OpInfo of an existing operator, please feel free to skip this test because the problem was pre-existing and file an issue. Otherwise, if you added a new operator, please read through the Composite Compliance section in aten/src/ATen/native/README.md for how to resolve this. ') from err
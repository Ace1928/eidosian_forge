import cmath
import math
import warnings
from collections import OrderedDict
from typing import Dict, Optional
import torch
import torch.backends.cudnn as cudnn
from ..nn.modules.utils import _list_with_default, _pair, _quadruple, _single, _triple
def register_all(mod):
    for name in dir(mod):
        v = getattr(mod, name)
        if callable(v) and (not _is_special_functional_bound_op(v)) and (v is not torch.no_grad) and (v is not torch.autocast):
            if name == '_segment_reduce':
                name = name[1:]
            _builtin_ops.append((v, 'aten::' + name))
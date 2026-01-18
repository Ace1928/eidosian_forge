import contextlib
from typing import Optional, Union, List, Set, Dict, Any
import warnings
from dataclasses import dataclass
import torch
import torchgen
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at,\
def alias_non_inplace_storage(arg, ret):
    if is_traceable_wrapper_subclass(arg) or is_traceable_wrapper_subclass(ret):
        ret_list = ret if isinstance(ret, list) else [ret]
        for r in ret_list:
            assert type(arg) == type(r), f'Called {str(func)} with input of type {type(arg)}\nand output of type {type(ret)}. But expected types to match.'
    with torch.utils._mode_utils.no_dispatch():
        meta_in_tls = torch._C._meta_in_tls_dispatch_include()
        torch._C._set_meta_in_tls_dispatch_include(True)
        try:
            if isinstance(ret, list):
                for r in ret:
                    torch.ops.aten.set_.source_Storage_storage_offset(r, arg.untyped_storage(), r.storage_offset(), r.shape)
            else:
                assert isinstance(ret, torch.Tensor), f'type: {type(ret)}'
                torch.ops.aten.set_.source_Storage_storage_offset(ret, arg.untyped_storage(), ret.storage_offset(), ret.shape)
        finally:
            torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)
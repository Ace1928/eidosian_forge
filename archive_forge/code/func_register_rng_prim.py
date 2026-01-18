from typing import Optional, Tuple
import torch
import torch.utils._pytree as pytree
from torch import _prims
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._prims_common import CUDARngStateHelper, make_contiguous_strides_for
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
from torch.types import _device, _dtype
def register_rng_prim(name, schema, impl_aten, impl_meta, doc, tags=None):
    rngprim.define(schema)
    rngprim_impl.impl(name, impl_aten)
    rngprim_meta_impl.impl(name, impl_meta)
    prim_packet = getattr(torch._ops.ops.rngprims, name)
    prim = prim_packet.default
    if tags:
        prim._tags = tags
    rngprim_autograd_impl.impl(name, backwards_not_supported(prim))
    for p in (prim_packet, prim):
        p.__doc__ = doc
        p.return_type = torch._prims_common.RETURN_TYPE.NEW
        p.schema = schema
        p.impl_aten = impl_aten
        p.prim_meta_impl = impl_meta
import torch
from functorch._C import dim as _C
from . import op_properties
from .batch_tensor import _enable_layers
from .tree_map import tree_flatten, tree_map
import operator
from functools import reduce
def t__getitem__(self, input):
    from . import _Tensor, Dim, DimensionBindError, DimList, Tensor, TensorLike
    is_simple = not isinstance(input, Dim) and (not isinstance(input, (tuple, list))) and (not (isinstance(input, TensorLike) and input.ndim == 0))
    if is_simple:
        if isinstance(self, _Tensor):
            return _Tensor.__torch_function__(_orig_getitem, None, (self, input))
        else:
            return _orig_getitem(self, input)
    if not isinstance(input, tuple):
        input = [input]
    else:
        input = list(input)
    dims_indexed = 0
    expanding_object = None
    dimlists = []
    for i, s in enumerate(input):
        if s is ... or (isinstance(s, DimList) and (not s.is_bound)):
            if expanding_object is not None:
                msg = f'at most one ... or unbound dimension list can exist in indexing list but found 2 at offsets {i} and {expanding_object}'
                raise DimensionBindError(msg)
            expanding_object = i
        if isinstance(s, DimList):
            dims_indexed += len(s) if s.is_bound else 0
            dimlists.append(i)
        elif s is not None and s is not ...:
            dims_indexed += 1
    ndim = self.ndim
    if dims_indexed > ndim:
        raise IndexError(f'at least {dims_indexed} indices were supplied but the tensor only has {ndim} dimensions.')
    if expanding_object is not None:
        expanding_ndims = ndim - dims_indexed
        obj = input[expanding_object]
        if obj is ...:
            input[expanding_object:expanding_object + 1] = [no_slice] * expanding_ndims
        else:
            obj.bind_len(expanding_ndims)
    for i in reversed(dimlists):
        input[i:i + 1] = input[i]
    dims_indexed = 0
    requires_view = False
    size = self.size()
    view_sizes = []
    dims_seen = dim_tracker()

    def add_dims(t):
        if not isinstance(t, _Tensor):
            return
        for d in t.dims:
            dims_seen.record(d)
    add_dims(self)
    dim_packs = []
    for i, idx in enumerate(input):
        if idx is None:
            input[i] = no_slice
            view_sizes.append(1)
            requires_view = True
        else:
            sz = size[dims_indexed]
            if isinstance(idx, Dim):
                idx.size = sz
                dims_seen.record(idx)
                view_sizes.append(sz)
            elif isinstance(idx, (tuple, list)) and idx and isinstance(idx[0], Dim):
                for d in idx:
                    dims_seen.record(idx)
                _bind_dims_to_size(sz, idx, f'offset {i}')
                view_sizes.extend((d.size for d in idx))
                requires_view = True
                dim_packs.append(i)
            else:
                add_dims(idx)
                view_sizes.append(sz)
            dims_indexed += 1
    if requires_view:
        self = self.view(*view_sizes)
    for i in reversed(dim_packs):
        input[i:i + 1] = input[i]
    if isinstance(self, _Tensor):
        ptensor_self, levels = (self._tensor, list(self._levels))
        input_it = iter(input)
        flat_inputs = [next(input_it) if isinstance(l, int) else l for l in levels]
        has_device = self._has_device
        to_pad = 0
    else:
        ptensor_self, flat_inputs = (self, input)
        to_pad = ptensor_self.ndim - len(flat_inputs)
        has_device = True
    result_levels = []
    index_levels = []
    tensor_insert_point = None
    to_expand = {}
    requires_getindex = False
    for i, inp in enumerate(flat_inputs):
        if isinstance(inp, Dim) and dims_seen[inp] == 1:
            flat_inputs[i] = no_slice
            result_levels.append(inp)
        elif isinstance(inp, TensorLike):
            requires_getindex = True
            if tensor_insert_point is None:
                tensor_insert_point = len(result_levels)
            ptensor, levels, _ = _tensor_levels(inp)
            to_expand[i] = levels
            flat_inputs[i] = ptensor
            for l in levels:
                if l not in index_levels:
                    index_levels.append(l)
        else:
            requires_getindex = True
            result_levels.append(0)
    if tensor_insert_point is not None:
        result_levels[tensor_insert_point:tensor_insert_point] = index_levels
    for i, levels in to_expand.items():
        flat_inputs[i] = _match_levels(flat_inputs[i], levels, index_levels)
    if requires_getindex:
        result = _orig_getitem(ptensor_self, flat_inputs)
    else:
        result = ptensor_self
    next_positional = -1
    if to_pad > 0:
        result_levels.extend([0] * to_pad)
    for i, r in enumerate(reversed(result_levels)):
        if isinstance(r, int):
            result_levels[-1 - i] = next_positional
            next_positional -= 1
    return Tensor.from_positional(result, result_levels, has_device)
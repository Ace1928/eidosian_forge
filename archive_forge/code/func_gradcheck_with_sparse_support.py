from typing import Optional, Tuple, List, Union, Any
import torch
from torch._C import _add_docstr, _sparse  # type: ignore[attr-defined]
from torch import Tensor
from .semi_structured import SparseSemiStructuredTensor, to_sparse_semi_structured
from typing import TYPE_CHECKING
def gradcheck_with_sparse_support(func, inputs, **kwargs):
    """
        Create gradcheck with support for sparse tensors.

        Same as :func:`torch.autograd.gradcheck` but with sparse tensors inputs and outputs support.
        """
    masked = kwargs.pop('masked', False)
    sparse_layouts = {torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}
    sparse_compressed_layouts = {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}
    sparse_block_layouts = {torch.sparse_bsr, torch.sparse_bsc}
    STRIDED_REPRESENTATION = '__STRIDED_REPRESENTATION__'

    def convert_to_strided_representation(args):
        """Convert differentiable non-strided tensors to a representation containing differentiable strided tensors."""
        if not isinstance(args, (list, tuple)):
            args = (args,)
        new_args: List[Any] = []
        for obj in args:
            if isinstance(obj, torch.Tensor) and obj.requires_grad and (obj.layout in sparse_layouts):
                d = dict(layout=obj.layout, shape=obj.shape)
                if not masked:
                    batch_dim = obj.ndim - obj.dense_dim() - obj.sparse_dim()
                    blocksize = obj.values().shape[batch_dim + 1:batch_dim + 3] if obj.layout in sparse_block_layouts else None
                    full_mask = torch.ones(obj.shape, device=obj.device, dtype=torch.bool).to_sparse(layout=obj.layout, blocksize=blocksize, dense_dim=obj.dense_dim())
                    obj = obj.to_dense().sparse_mask(full_mask)
                if obj.layout is torch.sparse_coo:
                    d.update(indices=obj._indices(), is_coalesced=obj.is_coalesced())
                    values = obj._values()
                elif obj.layout in {torch.sparse_csr, torch.sparse_bsr}:
                    d.update(compressed_indices=obj.crow_indices(), plain_indices=obj.col_indices())
                    values = obj.values()
                else:
                    d.update(compressed_indices=obj.ccol_indices(), plain_indices=obj.row_indices())
                    values = obj.values()
                new_args.extend((STRIDED_REPRESENTATION, d, values.requires_grad_(True)))
            else:
                new_args.append(obj)
        return tuple(new_args)

    def restore_from_strided_representation(args):
        """Restore non-strided differentiable tensosr from their strided representations."""
        new_args = []
        args = list(args)
        while args:
            a = args.pop(0)
            if a == STRIDED_REPRESENTATION:
                d, values = (args.pop(0), args.pop(0))
                if d['layout'] is torch.sparse_coo:
                    a = torch.sparse_coo_tensor(d['indices'], values, size=d['shape'], is_coalesced=d['is_coalesced'])
                elif d['layout'] in sparse_compressed_layouts:
                    a = torch.sparse_compressed_tensor(d['compressed_indices'], d['plain_indices'], values, size=d['shape'], layout=d['layout'])
                else:
                    raise NotImplementedError(f'conversion of {d['layout']} strided representation to tensor')
            new_args.append(a)
        return tuple(new_args)

    def func_wrapper(*args, **kwargs):
        restored_args = restore_from_strided_representation(args)
        outputs = func(*restored_args, **kwargs)
        strided_outputs = tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)
        strided_outputs = tuple((o.to_dense(masked_grad=masked) if isinstance(o, torch.Tensor) and o.requires_grad and (o.layout in sparse_layouts) else o for o in strided_outputs))
        return strided_outputs if isinstance(outputs, (list, tuple)) else strided_outputs[0]
    args = (func_wrapper, convert_to_strided_representation(inputs))
    return gradcheck(*args, **kwargs)
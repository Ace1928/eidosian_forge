import logging
import math
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from .tensor_utils import tensor_tree_map, tree_map
def chunk_layer(layer: Callable, inputs: Dict[str, Any], chunk_size: int, no_batch_dims: int, low_mem: bool=False, _out: Any=None, _add_into_out: bool=False) -> Any:
    """
    Implements the "chunking" procedure described in section 1.11.8.

    Layer outputs and inputs are assumed to be simple "pytrees," consisting only of (arbitrarily nested) lists, tuples,
    and dicts with torch.Tensor leaves.

    Args:
        layer:
            The layer to be applied chunk-wise
        inputs:
            A (non-nested) dictionary of keyworded inputs. All leaves must be tensors and must share the same batch
            dimensions.
        chunk_size:
            The number of sub-batches per chunk. If multiple batch dimensions are specified, a "sub-batch" is defined
            as a single indexing of all batch dimensions simultaneously (s.t. the number of sub-batches is the product
            of the batch dimensions).
        no_batch_dims:
            How many of the initial dimensions of each input tensor can be considered batch dimensions.
        low_mem:
            Avoids flattening potentially large input tensors. Unnecessary in most cases, and is ever so slightly
            slower than the default setting.
    Returns:
        The reassembled output of the layer on the inputs.
    """
    if not len(inputs) > 0:
        raise ValueError('Must provide at least one input')
    initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
    orig_batch_dims = tuple([max(s) for s in zip(*initial_dims)])

    def _prep_inputs(t: torch.Tensor) -> torch.Tensor:
        if not low_mem:
            if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
                t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
            t = t.reshape(-1, *t.shape[no_batch_dims:])
        else:
            t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
        return t
    prepped_inputs: Dict[str, Any] = tensor_tree_map(_prep_inputs, inputs)
    prepped_outputs = None
    if _out is not None:
        prepped_outputs = tensor_tree_map(lambda t: t.view([-1] + list(t.shape[no_batch_dims:])), _out)
    flat_batch_dim = 1
    for d in orig_batch_dims:
        flat_batch_dim *= d
    no_chunks = flat_batch_dim // chunk_size + (flat_batch_dim % chunk_size != 0)

    def _select_chunk(t: torch.Tensor) -> torch.Tensor:
        return t[i:i + chunk_size] if t.shape[0] != 1 else t
    i = 0
    out = prepped_outputs
    for _ in range(no_chunks):
        if not low_mem:
            select_chunk = _select_chunk
        else:
            select_chunk = partial(_chunk_slice, flat_start=i, flat_end=min(flat_batch_dim, i + chunk_size), no_batch_dims=len(orig_batch_dims))
        chunks: Dict[str, Any] = tensor_tree_map(select_chunk, prepped_inputs)
        output_chunk = layer(**chunks)
        if out is None:
            out = tensor_tree_map(lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:]), output_chunk)
        if isinstance(output_chunk, dict):

            def assign(d1: dict, d2: dict) -> None:
                for k, v in d1.items():
                    if isinstance(v, dict):
                        assign(v, d2[k])
                    elif _add_into_out:
                        v[i:i + chunk_size] += d2[k]
                    else:
                        v[i:i + chunk_size] = d2[k]
            assign(out, output_chunk)
        elif isinstance(output_chunk, tuple):
            for x1, x2 in zip(out, output_chunk):
                if _add_into_out:
                    x1[i:i + chunk_size] += x2
                else:
                    x1[i:i + chunk_size] = x2
        elif isinstance(output_chunk, torch.Tensor):
            if _add_into_out:
                out[i:i + chunk_size] += output_chunk
            else:
                out[i:i + chunk_size] = output_chunk
        else:
            raise ValueError('Not supported')
        i += chunk_size
    out = tensor_tree_map(lambda t: t.view(orig_batch_dims + t.shape[1:]), out)
    return out
from torch._functorch.vmap import (vmap_impl, _check_randomness_arg,
from torch._functorch.utils import exposed_in, argnums_t
import functools
def chunk_vmap(func: Callable, in_dims: in_dims_t=0, out_dims: out_dims_t=0, randomness: str='error', chunks=2) -> Callable:
    """
    chunk_vmap is the vectorizing map (vmap) using chunks of input data. It is a mix of vmap (which vectorizes
    everything) and map (which executes things sequentially). ``chunk_vmap`` vectorizes the input with number of
    chunks at a time. For more details about vectorizing map, see :func:`vmap`.

    .. note::
        Please use :func:`vmap` with ``chunk_size`` argument instead of this API.

    Args:
        func (function): A Python function that takes one or more arguments.
            Must return one or more Tensors.
        in_dims (int or nested structure): Specifies which dimension of the
            inputs should be mapped over. ``in_dims`` should have a
            structure like the inputs. If the ``in_dim`` for a particular
            input is None, then that indicates there is no map dimension.
            Default: 0.
        out_dims (int or Tuple[int]): Specifies where the mapped dimension
            should appear in the outputs. If ``out_dims`` is a Tuple, then
            it should have one element per output. Default: 0.
        randomness (str): Specifies whether the randomness in this
            vmap should be the same or different across batches. If 'different',
            the randomness for each batch will be different. If 'same', the
            randomness will be the same across batches. If 'error', any calls to
            random functions will error. Default: 'error'. WARNING: this flag
            only applies to random PyTorch operations and does not apply to
            Python's random module or numpy randomness.
        chunks (int): Number of chunks to use to split the input data. Default is 2.
            If equals to 1 then :func:`vmap` is called.

    Returns:
        Returns a new "batched" function. It takes the same inputs as
        ``func``, except each input has an extra dimension at the index
        specified by ``in_dims``. It takes returns the same outputs as
        ``func``, except each output has an extra dimension at the index
        specified by ``out_dims``.
    """
    _check_randomness_arg(randomness)
    if chunks == 1:
        return vmap(func, in_dims=in_dims, out_dims=out_dims, randomness=randomness)

    def _get_chunk_flat_args(flat_args_, flat_in_dims_, chunks_):
        flat_args_chunks = tuple((t.chunk(chunks_, dim=in_dim) if in_dim is not None else [t] * chunks_ for t, in_dim in zip(flat_args_, flat_in_dims_)))
        chunks_flat_args = zip(*flat_args_chunks)
        return chunks_flat_args

    @functools.wraps(func)
    def wrapped_with_chunks(*args, **kwargs):
        _check_out_dims_is_int_or_int_pytree(out_dims, func)
        _, flat_in_dims, flat_args, args_spec = _process_batched_inputs(in_dims, args, func)
        chunks_flat_args = _get_chunk_flat_args(flat_args, flat_in_dims, chunks)
        return _chunked_vmap(func, flat_in_dims, chunks_flat_args, args_spec, out_dims, randomness, **kwargs)
    return wrapped_with_chunks
from typing import Optional
import torch
import triton
import triton.language as tl
def scaled_index_add_fwd(x: torch.Tensor, index: torch.Tensor, source: torch.Tensor, scaling: Optional[torch.Tensor], alpha: float):
    if not (x.is_cuda and index.is_cuda and source.is_cuda):
        raise ValueError('The input tensor, the index tensor and the source tensor must be of type CUDA!')
    if not (x.ndim == 3 and source.ndim == 3):
        raise ValueError(f'The input and source must be three-dimensional (got {x.ndim} and {source.ndim})!')
    if not x.shape[1] == source.shape[1]:
        raise ValueError(f'The number of elements along dimension 1 of the input and source must be the same (got {(x.shape[1],)} and {(source.shape[1],)})!')
    if not x.shape[2] == source.shape[2]:
        raise ValueError(f'The number of elements along dimension 2 of the input and source must be the same (got {(x.shape[2],)} and {(source.shape[2],)})!')
    num_inp_indices, num_rows, num_cols = x.shape
    num_src_indices, num_rows, num_cols = source.shape
    if not num_inp_indices >= num_src_indices:
        raise ValueError(f'The number of elements along dimension 0 of the input must be larger than that of source (got {num_inp_indices} and {num_src_indices})!')
    if not index.shape[0] == num_src_indices:
        raise ValueError(f'The number of indices and source tensors must match (got {len(index)} and {len(source)})!')
    stride0, stride1, stride2 = (x.stride(0), x.stride(1), x.stride(2))
    if not (source.stride(0) == stride0 and source.stride(1) == stride1 and (source.stride(2) == stride2)):
        raise ValueError(f'The strides of the source and input tensors must match (got {source.stride(0)} vs. {stride0}, {source.stride(1)} vs. {stride1}, {source.stride(2)} vs. {stride2})!')
    if scaling is None:
        HAS_SCALING = False
    else:
        HAS_SCALING = True
        if not scaling.is_cuda:
            raise ValueError('The scaling tensor must be of type CUDA!')
        if not (scaling.ndim == 1 and scaling.numel() == num_cols):
            raise ValueError(f'The scaling tensor must be a 1-dimensional tensor (got {scaling.ndim}) and its size must be equal to the size of dimension 2 of source (got {scaling.numel()} vs. {num_cols}).')
        if not scaling.stride(0) == stride2:
            raise ValueError(f'The stride of scaling must match the stride2 of input (got {scaling.stride(0)} vs. {stride2})')
    if not index.ndim == 1:
        raise ValueError(f'The index must be one-dimensional (got {index.ndim})!')

    def grid(meta):
        return (triton.cdiv(num_src_indices, meta['BLOCK_SIZE_INDEX']), triton.cdiv(num_rows, meta['BLOCK_SIZE_ROW']), triton.cdiv(num_cols, meta['BLOCK_SIZE_COL']))
    scaled_index_add_fwd_kernel[grid](x, index, source, scaling, alpha, num_inp_indices, num_src_indices, num_rows, num_cols, x.stride(0), x.stride(1), x.stride(2), BLOCK_SIZE_INDEX=1, BLOCK_SIZE_ROW=1, BLOCK_SIZE_COL=512, HAS_SCALING=HAS_SCALING)
    return
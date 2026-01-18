import torch
import triton
import triton.language as tl
def index_select_cat_fwd(output: torch.Tensor, source: torch.Tensor, index: torch.Tensor):
    if not (source.is_cuda and index.is_cuda):
        raise ValueError('The index tensor and the source tensor must be of type CUDA!')
    if not source.ndim == 2:
        raise ValueError(f'Expected 2-dimensional tensor, got {source.ndim}.')
    if not index.ndim == 1:
        raise ValueError(f'Expected 1-dimensional tensor, got {index.ndim}.')
    num_rows, num_cols = source.shape
    num_indices = index.shape[0]
    if not num_indices < num_rows:
        raise ValueError('The number of indices cannot exceed the number of rows in the source matrix.')
    stride0, stride1 = (source.stride(0), source.stride(1))

    def grid(meta):
        return (triton.cdiv(num_indices, meta['BLOCK_SIZE_INDEX']), triton.cdiv(num_cols, meta['BLOCK_SIZE_COL']))
    index_select_cat_fwd_kernel[grid](output, source, index, num_indices, num_cols, stride0, stride1, BLOCK_SIZE_INDEX=1, BLOCK_SIZE_COL=512)
    return output
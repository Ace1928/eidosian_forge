from typing import Sequence
import torch
def split_tensor_along_last_dim(tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool=False) -> Sequence[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple((chunk.contiguous() for chunk in tensor_list))
    return tensor_list
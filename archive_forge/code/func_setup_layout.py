import random
import torch
def setup_layout(self, seq_len):
    """Create layout tensor for the given sequence length
        Arguments:
             seq_len: required: an integer determining number of attention heads of the layer.
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) for sparsity layout
                of all head; initialized with zero
        """
    if seq_len % self.block_size != 0:
        raise ValueError(f'Sequence Length, {seq_len}, needs to be dividable by Block size {self.block_size}!')
    num_blocks = seq_len // self.block_size
    layout = torch.zeros((self.num_heads, num_blocks, num_blocks), dtype=torch.int64)
    return layout
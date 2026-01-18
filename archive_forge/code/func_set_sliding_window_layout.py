import random
import torch
def set_sliding_window_layout(self, h, layout):
    """Sets sliding local attention layout used by the given head in the sparse attention.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout
                of all head in which local sliding window layout is set
        """
    num_blocks = layout.shape[1]
    if num_blocks < self.num_sliding_window_blocks:
        raise ValueError(f'Number of sliding window blocks, {self.num_sliding_window_blocks}, must be smaller\n                than overall number of blocks in a row, {num_blocks}!')
    w = self.num_sliding_window_blocks // 2
    for row in range(0, num_blocks):
        start = max(0, row - w)
        end = min(row + w + 1, num_blocks)
        layout[h, row, start:end] = 1
    return layout
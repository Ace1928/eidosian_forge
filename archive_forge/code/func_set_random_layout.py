import random
import torch
def set_random_layout(self, h, layout):
    """Sets random attention layout used by the given head in the sparse attention.
        Note) By default, it assumes there will be a unique random block layout for all heads; unless
        `different_layout_per_head` parameter is set in which each head can have a different random layout.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity
                layout of all head in which random layout is set
        """
    num_blocks = layout.shape[1]
    if num_blocks < self.num_random_blocks:
        raise ValueError(f'Number of random blocks, {self.num_random_blocks}, must be smaller than overall number\n                of blocks in a row, {num_blocks}!')
    for row in range(0, num_blocks):
        sample_range = range(0, num_blocks) if self.attention == 'bidirectional' else range(0, row + 1)
        rnd_cols = random.sample(sample_range, self.num_random_blocks)
        layout[h, row, rnd_cols] = 1
    return layout
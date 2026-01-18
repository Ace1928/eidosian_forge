import random
import torch
def check_and_propagate_first_head_layout(self, layout):
    """If all heads require same sparsity layout, it propagate first head layout to all heads
        Arguments:
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
             sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity
             layout of all head
        """
    if not self.different_layout_per_head:
        layout[1:self.num_heads, :, :] = layout[0, :, :]
    return layout
import random
import torch
def make_layout(self, seq_len):
    """Generates edited `Longformer` sparsity layout used by each head in the sparse attention.
        Arguments:
             seq_len: required: an integer determining number of attention heads of the layer.
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing `BSLongformer`
                sparsity layout of all head
        """
    layout = self.setup_layout(seq_len)
    for h in range(0, self.num_layout_heads):
        layout = self.set_sliding_window_layout(h, layout)
        layout = self.set_global_layout(h, layout)
    layout = self.check_and_propagate_first_head_layout(layout)
    return layout
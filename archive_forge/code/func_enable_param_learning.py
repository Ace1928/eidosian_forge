import torch
from torch.nn.parameter import Parameter
from typing import List
@torch.jit.export
def enable_param_learning(self):
    """Enable parameter learning over static observer estimates.

        Enables learning of quantization parameters and
        disables static observer estimates. Forward path returns fake quantized X.
        """
    self.toggle_qparam_learning(enabled=True).toggle_fake_quant(enabled=True).toggle_observer_update(enabled=False)
    return self
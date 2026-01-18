import math
from enum import Enum
from typing import Optional
import triton
import triton.language as tl
def get_triton_activation_bwd_kernel(activation: Optional[Activation]):
    return {Activation.ReLU: relu_grad, Activation.LeakyReLU: leaky_relu_grad, Activation.GeLU: gelu_grad, Activation.GeLUApprox: gelu_approx_grad, Activation.SquaredReLU: squared_relu_grad}[activation] if activation else None
from .core import MaskedTensor
def masked_tensor(data, mask, requires_grad=False):
    return MaskedTensor(data, mask, requires_grad)
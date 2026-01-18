import torch
def sparse_semi_structured_from_dense_cutlass(dense, compile=False):
    if compile:
        from torch._dynamo.utils import is_compile_supported
        if is_compile_supported(dense.device.type):
            kernel = torch.compile(_sparse_semi_structured_from_dense_cutlass)
            return kernel(dense)
    return _sparse_semi_structured_from_dense_cutlass(dense)
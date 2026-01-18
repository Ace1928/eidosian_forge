import torch._C._lazy
def set_reuse_ir(val: bool):
    """Set the config to reuse IR nodes for faster tracing"""
    torch._C._lazy._set_reuse_ir(val)
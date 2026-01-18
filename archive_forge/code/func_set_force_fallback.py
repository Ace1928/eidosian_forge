import torch._C._lazy
def set_force_fallback(configval):
    """Set the config used to force LTC fallback"""
    torch._C._lazy._set_force_fallback(configval)
import torch._C._lazy
def counter_names():
    """Retrieves all the currently active counter names."""
    return torch._C._lazy._counter_names()
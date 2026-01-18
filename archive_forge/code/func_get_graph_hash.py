import torch._C._lazy
import torch._C._lazy_ts_backend
def get_graph_hash(tensors):
    """Return the graph hash for the passed in lazy tensors"""
    return torch._C._lazy._get_graph_hash(tensors)
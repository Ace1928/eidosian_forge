import torch._C._lazy
import torch._C._lazy_ts_backend
def run_cached_graph(hash_str, graph_inputs):
    """Running the cached computation graph with the given inputs

    TODO: This API is currently ts backend specific. We are working on
    generalizing it to all backends including XLA.
    """
    return torch._C._lazy_ts_backend._run_cached_graph(hash_str, graph_inputs)
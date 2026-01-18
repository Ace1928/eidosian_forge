def get_static_sparse_quantized_mapping():
    import torch.ao.nn.sparse
    _static_sparse_quantized_mapping = {torch.nn.Linear: torch.ao.nn.sparse.quantized.Linear}
    return _static_sparse_quantized_mapping
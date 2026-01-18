def get_num_curiosity_nets(model_size, override=None):
    if override is not None:
        return override
    assert model_size in _ALLOWED_MODEL_DIMS
    num_curiosity_nets = {'nano': 8, 'micro': 8, 'mini': 16, 'XXS': 8, 'XS': 8, 'S': 8, 'M': 8, 'L': 8, 'XL': 8}
    return num_curiosity_nets[model_size]
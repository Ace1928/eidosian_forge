def get_num_dense_layers(model_size, override=None):
    if override is not None:
        return override
    assert model_size in _ALLOWED_MODEL_DIMS
    num_dense_layers = {'nano': 1, 'micro': 1, 'mini': 1, 'XXS': 1, 'XS': 1, 'S': 2, 'M': 3, 'L': 4, 'XL': 5}
    return num_dense_layers[model_size]
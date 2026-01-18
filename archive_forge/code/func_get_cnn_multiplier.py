def get_cnn_multiplier(model_size, override=None):
    if override is not None:
        return override
    assert model_size in _ALLOWED_MODEL_DIMS
    cnn_multipliers = {'nano': 2, 'micro': 4, 'mini': 8, 'XXS': 16, 'XS': 24, 'S': 32, 'M': 48, 'L': 64, 'XL': 96}
    return cnn_multipliers[model_size]